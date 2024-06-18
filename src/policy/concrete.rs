// SPDX-License-Identifier: CC0-1.0

//! Concrete Policies
//!

use core::{fmt, str};
#[cfg(feature = "std")]
use std::error;

use tapyrus::{absolute, Sequence};
#[cfg(feature = "compiler")]
use {
    crate::miniscript::ScriptContext, crate::policy::compiler,
    crate::policy::compiler::CompilerError, crate::Descriptor, crate::Miniscript,
};

use super::ENTAILMENT_MAX_TERMINALS;
use crate::expression::{self, FromTree};
use crate::iter::TreeLike;
use crate::miniscript::types::extra_props::TimelockInfo;
use crate::prelude::*;
use crate::sync::Arc;
#[cfg(all(doc, not(feature = "compiler")))]
use crate::Descriptor;
use crate::{errstr, AbsLockTime, Error, ForEachKey, MiniscriptKey, Translator};

/// Concrete policy which corresponds directly to a miniscript structure,
/// and whose disjunctions are annotated with satisfaction probabilities
/// to assist the compiler.
// Currently the vectors in And/Or are limited to two elements, this is a general miniscript thing
// not specific to rust-miniscript. Eventually we would like to extend these to be n-ary, but first
// we need to decide on a game plan for how to efficiently compile n-ary disjunctions
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Policy<Pk: MiniscriptKey> {
    /// Unsatisfiable.
    Unsatisfiable,
    /// Trivially satisfiable.
    Trivial,
    /// A public key which must sign to satisfy the descriptor.
    Key(Pk),
    /// An absolute locktime restriction.
    After(AbsLockTime),
    /// A relative locktime restriction.
    Older(Sequence),
    /// A SHA256 whose preimage must be provided to satisfy the descriptor.
    Sha256(Pk::Sha256),
    /// A SHA256d whose preimage must be provided to satisfy the descriptor.
    Hash256(Pk::Hash256),
    /// A RIPEMD160 whose preimage must be provided to satisfy the descriptor.
    Ripemd160(Pk::Ripemd160),
    /// A HASH160 whose preimage must be provided to satisfy the descriptor.
    Hash160(Pk::Hash160),
    /// A list of sub-policies, all of which must be satisfied.
    And(Vec<Arc<Policy<Pk>>>),
    /// A list of sub-policies, one of which must be satisfied, along with
    /// relative probabilities for each one.
    Or(Vec<(usize, Arc<Policy<Pk>>)>),
    /// A set of descriptors, satisfactions must be provided for `k` of them.
    Threshold(usize, Vec<Arc<Policy<Pk>>>),
}

impl<Pk> Policy<Pk>
where
    Pk: MiniscriptKey,
{
    /// Construct a `Policy::After` from `n`. Helper function equivalent to
    /// `Policy::After(absolute::LockTime::from_consensus(n))`.
    pub fn after(n: u32) -> Policy<Pk> {
        Policy::After(AbsLockTime::from(absolute::LockTime::from_consensus(n)))
    }

    /// Construct a `Policy::Older` from `n`. Helper function equivalent to
    /// `Policy::Older(Sequence::from_consensus(n))`.
    pub fn older(n: u32) -> Policy<Pk> { Policy::Older(Sequence::from_consensus(n)) }
}

/// Detailed error type for concrete policies.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum PolicyError {
    /// `And` fragments only support two args.
    NonBinaryArgAnd,
    /// `Or` fragments only support two args.
    NonBinaryArgOr,
    /// `Thresh` fragment can only have `1<=k<=n`.
    IncorrectThresh,
    /// `older` or `after` fragment can only have `n = 0`.
    ZeroTime,
    /// `after` fragment can only have `n < 2^31`.
    TimeTooFar,
    /// Semantic Policy Error: `And` `Or` fragments must take args: `k > 1`.
    InsufficientArgsforAnd,
    /// Semantic policy error: `And` `Or` fragments must take args: `k > 1`.
    InsufficientArgsforOr,
    /// Entailment max terminals exceeded.
    EntailmentMaxTerminals,
    /// Cannot lift policies that have a combination of height and timelocks.
    HeightTimelockCombination,
    /// Duplicate Public Keys.
    DuplicatePubKeys,
}

/// Descriptor context for [`Policy`] compilation into a [`Descriptor`].
pub enum DescriptorCtx {
    /// See docs for [`Descriptor::Bare`].
    Bare,
    /// See docs for [`Descriptor::Sh`].
    Sh,
}

impl fmt::Display for PolicyError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            PolicyError::NonBinaryArgAnd => {
                f.write_str("And policy fragment must take 2 arguments")
            }
            PolicyError::NonBinaryArgOr => f.write_str("Or policy fragment must take 2 arguments"),
            PolicyError::IncorrectThresh => {
                f.write_str("Threshold k must be greater than 0 and less than or equal to n 0<k<=n")
            }
            PolicyError::TimeTooFar => {
                f.write_str("Relative/Absolute time must be less than 2^31; n < 2^31")
            }
            PolicyError::ZeroTime => f.write_str("Time must be greater than 0; n > 0"),
            PolicyError::InsufficientArgsforAnd => {
                f.write_str("Semantic Policy 'And' fragment must have at least 2 args ")
            }
            PolicyError::InsufficientArgsforOr => {
                f.write_str("Semantic Policy 'Or' fragment must have at least 2 args ")
            }
            PolicyError::EntailmentMaxTerminals => {
                write!(f, "Policy entailment only supports {} terminals", ENTAILMENT_MAX_TERMINALS)
            }
            PolicyError::HeightTimelockCombination => {
                f.write_str("Cannot lift policies that have a heightlock and timelock combination")
            }
            PolicyError::DuplicatePubKeys => f.write_str("Policy contains duplicate keys"),
        }
    }
}

#[cfg(feature = "std")]
impl error::Error for PolicyError {
    fn cause(&self) -> Option<&dyn error::Error> {
        use self::PolicyError::*;

        match self {
            NonBinaryArgAnd
            | NonBinaryArgOr
            | IncorrectThresh
            | ZeroTime
            | TimeTooFar
            | InsufficientArgsforAnd
            | InsufficientArgsforOr
            | EntailmentMaxTerminals
            | HeightTimelockCombination
            | DuplicatePubKeys => None,
        }
    }
}

impl<Pk: MiniscriptKey> Policy<Pk> {
    /// Compiles the [`Policy`] into `desc_ctx` [`Descriptor`]
    ///
    /// # NOTE:
    ///
    /// It is **not recommended** to use policy as a stable identifier for a miniscript. You should
    /// use the policy compiler once, and then use the miniscript output as a stable identifier. See
    /// the compiler document in [`doc/compiler.md`] for more details.
    #[cfg(feature = "compiler")]
    pub fn compile_to_descriptor<Ctx: ScriptContext>(
        &self,
        desc_ctx: DescriptorCtx,
    ) -> Result<Descriptor<Pk>, Error> {
        self.is_valid()?;
        match self.is_safe_nonmalleable() {
            (false, _) => Err(Error::from(CompilerError::TopLevelNonSafe)),
            (_, false) => Err(Error::from(CompilerError::ImpossibleNonMalleableCompilation)),
            _ => match desc_ctx {
                DescriptorCtx::Bare => Descriptor::new_bare(compiler::best_compilation(self)?),
                DescriptorCtx::Sh => Descriptor::new_sh(compiler::best_compilation(self)?),
            },
        }
    }

    /// Compiles the descriptor into an optimized `Miniscript` representation.
    ///
    /// # NOTE:
    ///
    /// It is **not recommended** to use policy as a stable identifier for a miniscript. You should
    /// use the policy compiler once, and then use the miniscript output as a stable identifier. See
    /// the compiler document in doc/compiler.md for more details.
    #[cfg(feature = "compiler")]
    pub fn compile<Ctx: ScriptContext>(&self) -> Result<Miniscript<Pk, Ctx>, CompilerError> {
        self.is_valid()?;
        match self.is_safe_nonmalleable() {
            (false, _) => Err(CompilerError::TopLevelNonSafe),
            (_, false) => Err(CompilerError::ImpossibleNonMalleableCompilation),
            _ => compiler::best_compilation(self),
        }
    }
}

#[cfg(feature = "compiler")]
impl<Pk: MiniscriptKey> Policy<Pk> {}

impl<Pk: MiniscriptKey> ForEachKey<Pk> for Policy<Pk> {
    fn for_each_key<'a, F: FnMut(&'a Pk) -> bool>(&'a self, mut pred: F) -> bool {
        self.pre_order_iter().all(|policy| match policy {
            Policy::Key(ref pk) => pred(pk),
            _ => true,
        })
    }
}

impl<Pk: MiniscriptKey> Policy<Pk> {
    /// Converts a policy using one kind of public key to another type of public key.
    ///
    /// For example usage please see [`crate::policy::semantic::Policy::translate_pk`].
    pub fn translate_pk<Q, E, T>(&self, t: &mut T) -> Result<Policy<Q>, E>
    where
        T: Translator<Pk, Q, E>,
        Q: MiniscriptKey,
    {
        use Policy::*;

        let mut translated = vec![];
        for data in self.post_order_iter() {
            let child_n = |n| Arc::clone(&translated[data.child_indices[n]]);

            let new_policy = match data.node {
                Unsatisfiable => Unsatisfiable,
                Trivial => Trivial,
                Key(ref pk) => t.pk(pk).map(Key)?,
                Sha256(ref h) => t.sha256(h).map(Sha256)?,
                Hash256(ref h) => t.hash256(h).map(Hash256)?,
                Ripemd160(ref h) => t.ripemd160(h).map(Ripemd160)?,
                Hash160(ref h) => t.hash160(h).map(Hash160)?,
                Older(ref n) => Older(*n),
                After(ref n) => After(*n),
                And(ref subs) => And((0..subs.len()).map(child_n).collect()),
                Or(ref subs) => Or(subs
                    .iter()
                    .enumerate()
                    .map(|(i, (prob, _))| (*prob, child_n(i)))
                    .collect()),
                Threshold(ref k, ref subs) => Threshold(*k, (0..subs.len()).map(child_n).collect()),
            };
            translated.push(Arc::new(new_policy));
        }
        // Unwrap is ok because we know we processed at least one node.
        let root_node = translated.pop().unwrap();
        // Unwrap is ok because we know `root_node` is the only strong reference.
        Ok(Arc::try_unwrap(root_node).unwrap())
    }

    /// Translates `Concrete::Key(key)` to `Concrete::Unsatisfiable` when extracting `TapKey`.
    pub fn translate_unsatisfiable_pk(self, key: &Pk) -> Policy<Pk> {
        use Policy::*;

        let mut translated = vec![];
        for data in Arc::new(self).post_order_iter() {
            let child_n = |n| Arc::clone(&translated[data.child_indices[n]]);

            let new_policy = match data.node.as_ref() {
                Policy::Key(ref k) if k.clone() == *key => Some(Policy::Unsatisfiable),
                And(ref subs) => Some(And((0..subs.len()).map(child_n).collect())),
                Or(ref subs) => Some(Or(subs
                    .iter()
                    .enumerate()
                    .map(|(i, (prob, _))| (*prob, child_n(i)))
                    .collect())),
                Threshold(k, ref subs) => {
                    Some(Threshold(*k, (0..subs.len()).map(child_n).collect()))
                }
                _ => None,
            };
            match new_policy {
                Some(new_policy) => translated.push(Arc::new(new_policy)),
                None => translated.push(Arc::clone(&data.node)),
            }
        }
        // Ok to unwrap because we know we processed at least one node.
        let root_node = translated.pop().unwrap();
        // Ok to unwrap because we know `root_node` is the only strong reference.
        Arc::try_unwrap(root_node).unwrap()
    }

    /// Gets all keys in the policy.
    pub fn keys(&self) -> Vec<&Pk> {
        self.pre_order_iter()
            .filter_map(|policy| match policy {
                Policy::Key(ref pk) => Some(pk),
                _ => None,
            })
            .collect()
    }

    /// Checks whether the policy contains duplicate public keys.
    pub fn check_duplicate_keys(&self) -> Result<(), PolicyError> {
        let pks = self.keys();
        let pks_len = pks.len();
        let unique_pks_len = pks.into_iter().collect::<BTreeSet<_>>().len();

        if pks_len > unique_pks_len {
            Err(PolicyError::DuplicatePubKeys)
        } else {
            Ok(())
        }
    }

    /// Checks whether the given concrete policy contains a combination of
    /// timelocks and heightlocks.
    ///
    /// # Returns
    ///
    /// Returns an error if there is at least one satisfaction that contains
    /// a combination of heightlock and timelock.
    pub fn check_timelocks(&self) -> Result<(), PolicyError> {
        let aggregated_timelock_info = self.timelock_info();
        if aggregated_timelock_info.contains_combination {
            Err(PolicyError::HeightTimelockCombination)
        } else {
            Ok(())
        }
    }

    /// Processes `Policy` using `post_order_iter`, creates a `TimelockInfo` for each `Nullary` node
    /// and combines them together for `Nary` nodes.
    ///
    /// # Returns
    ///
    /// A single `TimelockInfo` that is the combination of all others after processing each node.
    fn timelock_info(&self) -> TimelockInfo {
        use Policy::*;

        let mut infos = vec![];
        for data in Arc::new(self).post_order_iter() {
            let info_for_child_n = |n| infos[data.child_indices[n]];

            let info = match data.node {
                Policy::After(ref t) => TimelockInfo {
                    csv_with_height: false,
                    csv_with_time: false,
                    cltv_with_height: absolute::LockTime::from(*t).is_block_height(),
                    cltv_with_time: absolute::LockTime::from(*t).is_block_time(),
                    contains_combination: false,
                },
                Policy::Older(ref t) => TimelockInfo {
                    csv_with_height: t.is_height_locked(),
                    csv_with_time: t.is_time_locked(),
                    cltv_with_height: false,
                    cltv_with_time: false,
                    contains_combination: false,
                },
                And(ref subs) => {
                    let iter = (0..subs.len()).map(info_for_child_n);
                    TimelockInfo::combine_threshold(subs.len(), iter)
                }
                Or(ref subs) => {
                    let iter = (0..subs.len()).map(info_for_child_n);
                    TimelockInfo::combine_threshold(1, iter)
                }
                Threshold(ref k, subs) => {
                    let iter = (0..subs.len()).map(info_for_child_n);
                    TimelockInfo::combine_threshold(*k, iter)
                }
                _ => TimelockInfo::default(),
            };
            infos.push(info);
        }
        // Ok to unwrap, we had to have visited at least one node.
        infos.pop().unwrap()
    }

    /// This returns whether the given policy is valid or not. It maybe possible that the policy
    /// contains Non-two argument `and`, `or` or a `0` arg thresh.
    /// Validity condition also checks whether there is a possible satisfaction
    /// combination of timelocks and heightlocks
    pub fn is_valid(&self) -> Result<(), PolicyError> {
        use Policy::*;

        self.check_timelocks()?;
        self.check_duplicate_keys()?;

        for policy in self.pre_order_iter() {
            match *policy {
                After(n) => {
                    if n == absolute::LockTime::ZERO.into() {
                        return Err(PolicyError::ZeroTime);
                    } else if n.to_u32() > 2u32.pow(31) {
                        return Err(PolicyError::TimeTooFar);
                    }
                }
                Older(n) => {
                    if n == Sequence::ZERO {
                        return Err(PolicyError::ZeroTime);
                    } else if n.to_consensus_u32() > 2u32.pow(31) {
                        return Err(PolicyError::TimeTooFar);
                    }
                }
                And(ref subs) => {
                    if subs.len() != 2 {
                        return Err(PolicyError::NonBinaryArgAnd);
                    }
                }
                Or(ref subs) => {
                    if subs.len() != 2 {
                        return Err(PolicyError::NonBinaryArgOr);
                    }
                }
                Threshold(k, ref subs) => {
                    if k == 0 || k > subs.len() {
                        return Err(PolicyError::IncorrectThresh);
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Checks if any possible compilation of the policy could be compiled
    /// as non-malleable and safe.
    ///
    /// # Returns
    ///
    /// Returns a tuple `(safe, non-malleable)` to avoid the fact that
    /// non-malleability depends on safety and we would like to cache results.
    pub fn is_safe_nonmalleable(&self) -> (bool, bool) {
        use Policy::*;

        let mut acc = vec![];
        for data in Arc::new(self).post_order_iter() {
            let acc_for_child_n = |n| acc[data.child_indices[n]];

            let new = match data.node {
                Unsatisfiable | Trivial | Key(_) => (true, true),
                Sha256(_) | Hash256(_) | Ripemd160(_) | Hash160(_) | After(_) | Older(_) => {
                    (false, true)
                }
                And(ref subs) => {
                    let (atleast_one_safe, all_non_mall) = (0..subs.len())
                        .map(acc_for_child_n)
                        .fold((false, true), |acc, x: (bool, bool)| (acc.0 || x.0, acc.1 && x.1));
                    (atleast_one_safe, all_non_mall)
                }
                Or(ref subs) => {
                    let (all_safe, atleast_one_safe, all_non_mall) = (0..subs.len())
                        .map(acc_for_child_n)
                        .fold((true, false, true), |acc, x| {
                            (acc.0 && x.0, acc.1 || x.0, acc.2 && x.1)
                        });
                    (all_safe, atleast_one_safe && all_non_mall)
                }
                Threshold(k, ref subs) => {
                    let (safe_count, non_mall_count) = (0..subs.len()).map(acc_for_child_n).fold(
                        (0, 0),
                        |(safe_count, non_mall_count), (safe, non_mall)| {
                            (safe_count + safe as usize, non_mall_count + non_mall as usize)
                        },
                    );
                    (
                        safe_count >= (subs.len() - k + 1),
                        non_mall_count == subs.len() && safe_count >= (subs.len() - k),
                    )
                }
            };
            acc.push(new);
        }
        // Ok to unwrap because we know we processed at least one node.
        acc.pop().unwrap()
    }
}

impl<Pk: MiniscriptKey> fmt::Debug for Policy<Pk> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Policy::Unsatisfiable => f.write_str("UNSATISFIABLE()"),
            Policy::Trivial => f.write_str("TRIVIAL()"),
            Policy::Key(ref pk) => write!(f, "pk({:?})", pk),
            Policy::After(n) => write!(f, "after({})", n),
            Policy::Older(n) => write!(f, "older({})", n),
            Policy::Sha256(ref h) => write!(f, "sha256({})", h),
            Policy::Hash256(ref h) => write!(f, "hash256({})", h),
            Policy::Ripemd160(ref h) => write!(f, "ripemd160({})", h),
            Policy::Hash160(ref h) => write!(f, "hash160({})", h),
            Policy::And(ref subs) => {
                f.write_str("and(")?;
                if !subs.is_empty() {
                    write!(f, "{:?}", subs[0])?;
                    for sub in &subs[1..] {
                        write!(f, ",{:?}", sub)?;
                    }
                }
                f.write_str(")")
            }
            Policy::Or(ref subs) => {
                f.write_str("or(")?;
                if !subs.is_empty() {
                    write!(f, "{}@{:?}", subs[0].0, subs[0].1)?;
                    for sub in &subs[1..] {
                        write!(f, ",{}@{:?}", sub.0, sub.1)?;
                    }
                }
                f.write_str(")")
            }
            Policy::Threshold(k, ref subs) => {
                write!(f, "thresh({}", k)?;
                for sub in subs {
                    write!(f, ",{:?}", sub)?;
                }
                f.write_str(")")
            }
        }
    }
}

impl<Pk: MiniscriptKey> fmt::Display for Policy<Pk> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Policy::Unsatisfiable => f.write_str("UNSATISFIABLE"),
            Policy::Trivial => f.write_str("TRIVIAL"),
            Policy::Key(ref pk) => write!(f, "pk({})", pk),
            Policy::After(n) => write!(f, "after({})", n),
            Policy::Older(n) => write!(f, "older({})", n),
            Policy::Sha256(ref h) => write!(f, "sha256({})", h),
            Policy::Hash256(ref h) => write!(f, "hash256({})", h),
            Policy::Ripemd160(ref h) => write!(f, "ripemd160({})", h),
            Policy::Hash160(ref h) => write!(f, "hash160({})", h),
            Policy::And(ref subs) => {
                f.write_str("and(")?;
                if !subs.is_empty() {
                    write!(f, "{}", subs[0])?;
                    for sub in &subs[1..] {
                        write!(f, ",{}", sub)?;
                    }
                }
                f.write_str(")")
            }
            Policy::Or(ref subs) => {
                f.write_str("or(")?;
                if !subs.is_empty() {
                    write!(f, "{}@{}", subs[0].0, subs[0].1)?;
                    for sub in &subs[1..] {
                        write!(f, ",{}@{}", sub.0, sub.1)?;
                    }
                }
                f.write_str(")")
            }
            Policy::Threshold(k, ref subs) => {
                write!(f, "thresh({}", k)?;
                for sub in subs {
                    write!(f, ",{}", sub)?;
                }
                f.write_str(")")
            }
        }
    }
}

impl_from_str!(
    Policy<Pk>,
    type Err = Error;,
    fn from_str(s: &str) -> Result<Policy<Pk>, Error> {
        expression::check_valid_chars(s)?;

        let tree = expression::Tree::from_str(s)?;
        let policy: Policy<Pk> = FromTree::from_tree(&tree)?;
        policy.check_timelocks()?;
        Ok(policy)
    }
);

serde_string_impl_pk!(Policy, "a miniscript concrete policy");

#[rustfmt::skip]
impl_block_str!(
    Policy<Pk>,
    /// Helper function for `from_tree` to parse subexpressions with
    /// names of the form x@y
    fn from_tree_prob(top: &expression::Tree, allow_prob: bool,)
        -> Result<(usize, Policy<Pk>), Error>
    {
        let frag_prob;
        let frag_name;
        let mut name_split = top.name.split('@');
        match (name_split.next(), name_split.next(), name_split.next()) {
            (None, _, _) => {
                frag_prob = 1;
                frag_name = "";
            }
            (Some(name), None, _) => {
                frag_prob = 1;
                frag_name = name;
            }
            (Some(prob), Some(name), None) => {
                if !allow_prob {
                    return Err(Error::AtOutsideOr(top.name.to_owned()));
                }
                frag_prob = expression::parse_num(prob)? as usize;
                frag_name = name;
            }
            (Some(_), Some(_), Some(_)) => {
                return Err(Error::MultiColon(top.name.to_owned()));
            }
        }
        match (frag_name, top.args.len() as u32) {
            ("UNSATISFIABLE", 0) => Ok(Policy::Unsatisfiable),
            ("TRIVIAL", 0) => Ok(Policy::Trivial),
            ("pk", 1) => expression::terminal(&top.args[0], |pk| Pk::from_str(pk).map(Policy::Key)),
            ("after", 1) => {
                let num = expression::terminal(&top.args[0], expression::parse_num)?;
                if num > 2u32.pow(31) {
                    return Err(Error::PolicyError(PolicyError::TimeTooFar));
                } else if num == 0 {
                    return Err(Error::PolicyError(PolicyError::ZeroTime));
                }
                Ok(Policy::after(num))
            }
            ("older", 1) => {
                let num = expression::terminal(&top.args[0], expression::parse_num)?;
                if num > 2u32.pow(31) {
                    return Err(Error::PolicyError(PolicyError::TimeTooFar));
                } else if num == 0 {
                    return Err(Error::PolicyError(PolicyError::ZeroTime));
                }
                Ok(Policy::older(num))
            }
            ("sha256", 1) => expression::terminal(&top.args[0], |x| {
                <Pk::Sha256 as core::str::FromStr>::from_str(x).map(Policy::Sha256)
            }),
            ("hash256", 1) => expression::terminal(&top.args[0], |x| {
                <Pk::Hash256 as core::str::FromStr>::from_str(x).map(Policy::Hash256)
            }),
            ("ripemd160", 1) => expression::terminal(&top.args[0], |x| {
                <Pk::Ripemd160 as core::str::FromStr>::from_str(x).map(Policy::Ripemd160)
            }),
            ("hash160", 1) => expression::terminal(&top.args[0], |x| {
                <Pk::Hash160 as core::str::FromStr>::from_str(x).map(Policy::Hash160)
            }),
            ("and", _) => {
                if top.args.len() != 2 {
                    return Err(Error::PolicyError(PolicyError::NonBinaryArgAnd));
                }
                let mut subs = Vec::with_capacity(top.args.len());
                for arg in &top.args {
                    subs.push(Arc::new(Policy::from_tree(arg)?));
                }
                Ok(Policy::And(subs))
            }
            ("or", _) => {
                if top.args.len() != 2 {
                    return Err(Error::PolicyError(PolicyError::NonBinaryArgOr));
                }
                let mut subs = Vec::with_capacity(top.args.len());
                for arg in &top.args {
                    subs.push(Policy::from_tree_prob(arg, true)?);
                }
                Ok(Policy::Or(subs.into_iter().map(|(prob, sub)| (prob, Arc::new(sub))).collect()))
            }
            ("thresh", nsubs) => {
                if top.args.is_empty() || !top.args[0].args.is_empty() {
                    return Err(Error::PolicyError(PolicyError::IncorrectThresh));
                }

                let thresh = expression::parse_num(top.args[0].name)?;
                if thresh >= nsubs || thresh == 0 {
                    return Err(Error::PolicyError(PolicyError::IncorrectThresh));
                }

                let mut subs = Vec::with_capacity(top.args.len() - 1);
                for arg in &top.args[1..] {
                    subs.push(Policy::from_tree(arg)?);
                }
                Ok(Policy::Threshold(thresh as usize, subs.into_iter().map(Arc::new).collect()))
            }
            _ => Err(errstr(top.name)),
        }
        .map(|res| (frag_prob, res))
    }
);

impl_from_tree!(
    Policy<Pk>,
    fn from_tree(top: &expression::Tree) -> Result<Policy<Pk>, Error> {
        Policy::from_tree_prob(top, false).map(|(_, result)| result)
    }
);

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;

    #[test]
    fn for_each_key_count_keys() {
        let liquid_pol = Policy::<String>::from_str(
            "or(and(older(4096),thresh(2,pk(A),pk(B),pk(C))),thresh(11,pk(F1),pk(F2),pk(F3),pk(F4),pk(F5),pk(F6),pk(F7),pk(F8),pk(F9),pk(F10),pk(F11),pk(F12),pk(F13),pk(F14)))").unwrap();
        let mut count = 0;
        assert!(liquid_pol.for_each_key(|_| {
            count += 1;
            true
        }));
        assert_eq!(count, 17);
    }

    #[test]
    fn for_each_key_fails_predicate() {
        let policy =
            Policy::<String>::from_str("or(and(pk(key0),pk(key1)),pk(oddnamedkey))").unwrap();
        assert!(!policy.for_each_key(|k| k.starts_with("key")));
    }

    #[test]
    fn tranaslate_pk() {
        pub struct TestTranslator;
        impl Translator<String, String, ()> for TestTranslator {
            fn pk(&mut self, pk: &String) -> Result<String, ()> {
                let new = format!("NEW-{}", pk);
                Ok(new.to_string())
            }
            fn sha256(&mut self, hash: &String) -> Result<String, ()> { Ok(hash.to_string()) }
            fn hash256(&mut self, hash: &String) -> Result<String, ()> { Ok(hash.to_string()) }
            fn ripemd160(&mut self, hash: &String) -> Result<String, ()> { Ok(hash.to_string()) }
            fn hash160(&mut self, hash: &String) -> Result<String, ()> { Ok(hash.to_string()) }
        }
        let policy = Policy::<String>::from_str("or(and(pk(A),pk(B)),pk(C))").unwrap();
        let mut t = TestTranslator;

        let want = Policy::<String>::from_str("or(and(pk(NEW-A),pk(NEW-B)),pk(NEW-C))").unwrap();
        let got = policy
            .translate_pk(&mut t)
            .expect("failed to translate keys");

        assert_eq!(got, want);
    }

    #[test]
    fn translate_unsatisfiable_pk() {
        let policy = Policy::<String>::from_str("or(and(pk(A),pk(B)),pk(C))").unwrap();

        let want = Policy::<String>::from_str("or(and(pk(A),UNSATISFIABLE),pk(C))").unwrap();
        let got = policy.translate_unsatisfiable_pk(&"B".to_string());

        assert_eq!(got, want);
    }

    #[test]
    fn keys() {
        let policy = Policy::<String>::from_str("or(and(pk(A),pk(B)),pk(C))").unwrap();

        let want = vec!["A", "B", "C"];
        let got = policy.keys();

        assert_eq!(got, want);
    }

    #[test]
    #[should_panic]
    fn check_timelocks() {
        // This implicitly tests the check_timelocks API (has height and time locks).
        let _ = Policy::<String>::from_str("and(after(10),after(500000000))").unwrap();
    }
}
