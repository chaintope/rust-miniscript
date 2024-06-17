// SPDX-License-Identifier: CC0-1.0

//! # P2SH Descriptors
//!
//! Implementation of p2sh descriptors. Contains the implementation
//! of sh, wrapped fragments for sh which include wsh, sortedmulti
//! sh(miniscript), and sh(wpkh)
//!

use core::fmt;

use tapyrus::{Address, Network, ScriptBuf};

use super::checksum::verify_checksum;
use super::SortedMultiVec;
use crate::descriptor::{write_descriptor, DefiniteDescriptorKey};
use crate::expression::{self, FromTree};
use crate::miniscript::context::ScriptContext;
use crate::miniscript::satisfy::{Placeholder, Satisfaction};
use crate::plan::AssetProvider;
use crate::policy::{semantic, Liftable};
use crate::prelude::*;
use crate::util::{varint_len, witness_to_scriptsig};
use crate::{
    push_opcode_size, Error, ForEachKey, Legacy, Miniscript, MiniscriptKey, Satisfier,
    ToPublicKey, TranslateErr, TranslatePk, Translator,
};

/// A Legacy p2sh Descriptor
#[derive(Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct Sh<Pk: MiniscriptKey> {
    /// underlying miniscript
    inner: ShInner<Pk>,
}

/// Sh Inner
#[derive(Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum ShInner<Pk: MiniscriptKey> {
    /// Inner Sorted Multi
    SortedMulti(SortedMultiVec<Pk, Legacy>),
    /// p2sh miniscript
    Ms(Miniscript<Pk, Legacy>),
}

impl<Pk: MiniscriptKey> Liftable<Pk> for Sh<Pk> {
    fn lift(&self) -> Result<semantic::Policy<Pk>, Error> {
        match self.inner {
            ShInner::SortedMulti(ref smv) => smv.lift(),
            ShInner::Ms(ref ms) => ms.lift(),
        }
    }
}

impl<Pk: MiniscriptKey> fmt::Debug for Sh<Pk> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.inner {
            ShInner::SortedMulti(ref smv) => write!(f, "sh({:?})", smv),
            ShInner::Ms(ref ms) => write!(f, "sh({:?})", ms),
        }
    }
}

impl<Pk: MiniscriptKey> fmt::Display for Sh<Pk> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.inner {
            ShInner::SortedMulti(ref smv) => write_descriptor!(f, "sh({})", smv),
            ShInner::Ms(ref ms) => write_descriptor!(f, "sh({})", ms),
        }
    }
}

impl_from_tree!(
    Sh<Pk>,
    fn from_tree(top: &expression::Tree) -> Result<Self, Error> {
        if top.name == "sh" && top.args.len() == 1 {
            let top = &top.args[0];
            let inner = match top.name {
                "sortedmulti" => ShInner::SortedMulti(SortedMultiVec::from_tree(top)?),
                _ => {
                    let sub = Miniscript::from_tree(top)?;
                    Legacy::top_level_checks(&sub)?;
                    ShInner::Ms(sub)
                }
            };
            Ok(Sh { inner })
        } else {
            Err(Error::Unexpected(format!(
                "{}({} args) while parsing sh descriptor",
                top.name,
                top.args.len(),
            )))
        }
    }
);

impl_from_str!(
    Sh<Pk>,
    type Err = Error;,
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let desc_str = verify_checksum(s)?;
        let top = expression::Tree::from_str(desc_str)?;
        Self::from_tree(&top)
    }
);

impl<Pk: MiniscriptKey> Sh<Pk> {
    /// Get the Inner
    pub fn into_inner(self) -> ShInner<Pk> { self.inner }

    /// Get a reference to inner
    pub fn as_inner(&self) -> &ShInner<Pk> { &self.inner }

    /// Create a new p2sh descriptor with the raw miniscript
    pub fn new(ms: Miniscript<Pk, Legacy>) -> Result<Self, Error> {
        // do the top-level checks
        Legacy::top_level_checks(&ms)?;
        Ok(Self { inner: ShInner::Ms(ms) })
    }

    /// Create a new p2sh sortedmulti descriptor with threshold `k`
    /// and Vec of `pks`.
    pub fn new_sortedmulti(k: usize, pks: Vec<Pk>) -> Result<Self, Error> {
        // The context checks will be carried out inside new function for
        // sortedMultiVec
        Ok(Self { inner: ShInner::SortedMulti(SortedMultiVec::new(k, pks)?) })
    }

    /// Checks whether the descriptor is safe.
    pub fn sanity_check(&self) -> Result<(), Error> {
        match self.inner {
            ShInner::SortedMulti(ref smv) => smv.sanity_check()?,
            ShInner::Ms(ref ms) => ms.sanity_check()?,
        }
        Ok(())
    }

    /// Computes an upper bound on the difference between a non-satisfied
    /// `TxIn`'s `segwit_weight` and a satisfied `TxIn`'s `segwit_weight`
    ///
    /// Since this method uses `segwit_weight` instead of `legacy_weight`,
    /// if you want to include only legacy inputs in your transaction,
    /// you should remove 1WU from each input's `max_weight_to_satisfy`
    /// for a more accurate estimate.
    ///
    /// Assumes all ec-signatures are 73 bytes, including push opcode and
    /// sighash suffix.
    ///
    /// # Errors
    /// When the descriptor is impossible to safisfy (ex: sh(OP_FALSE)).
    pub fn max_weight_to_satisfy(&self) -> Result<usize, Error> {
        let (scriptsig_size, witness_size) = match self.inner {
            ShInner::SortedMulti(ref smv) => {
                let ss = smv.script_size();
                let ps = push_opcode_size(ss);
                let scriptsig_size = ps + ss + smv.max_satisfaction_size();
                (scriptsig_size, 0)
            }
            ShInner::Ms(ref ms) => {
                let ss = ms.script_size();
                let ps = push_opcode_size(ss);
                let scriptsig_size = ps + ss + ms.max_satisfaction_size()?;
                (scriptsig_size, 0)
            }
        };

        // scriptSigLen varint difference between non-satisfied (0) and satisfied
        let scriptsig_varint_diff = varint_len(scriptsig_size) - varint_len(0);

        Ok(4 * (scriptsig_varint_diff + scriptsig_size) + witness_size)
    }

    /// Computes an upper bound on the weight of a satisfying witness to the
    /// transaction.
    ///
    /// Assumes all ECDSA signatures are 73 bytes, including push opcode and
    /// sighash suffix. Includes the weight of the VarInts encoding the
    /// scriptSig and witness stack length.
    ///
    /// # Errors
    /// When the descriptor is impossible to safisfy (ex: sh(OP_FALSE)).
    #[deprecated(note = "use max_weight_to_satisfy instead")]
    #[allow(deprecated)]
    pub fn max_satisfaction_weight(&self) -> Result<usize, Error> {
        Ok(match self.inner {
            ShInner::SortedMulti(ref smv) => {
                let ss = smv.script_size();
                let ps = push_opcode_size(ss);
                let scriptsig_len = ps + ss + smv.max_satisfaction_size();
                4 * (varint_len(scriptsig_len) + scriptsig_len)
            }
            ShInner::Ms(ref ms) => {
                let ss = ms.script_size();
                let ps = push_opcode_size(ss);
                let scriptsig_len = ps + ss + ms.max_satisfaction_size()?;
                4 * (varint_len(scriptsig_len) + scriptsig_len)
            }
        })
    }
}

impl<Pk: MiniscriptKey + ToPublicKey> Sh<Pk> {
    /// Obtains the corresponding script pubkey for this descriptor.
    pub fn script_pubkey(&self) -> ScriptBuf {
        match self.inner {
            ShInner::SortedMulti(ref smv) => smv.encode().to_p2sh(),
            ShInner::Ms(ref ms) => ms.encode().to_p2sh(),
        }
    }

    /// Obtains the corresponding address for this descriptor.
    pub fn address(&self, network: Network) -> Address {
        let addr = self.address_fallible(network);

        // Size is checked in `check_global_consensus_validity`.
        assert!(addr.is_ok());
        addr.expect("only fails if size > MAX_SCRIPT_ELEMENT_SIZE")
    }

    fn address_fallible(&self, network: Network) -> Result<Address, Error> {
        let script = match self.inner {
            ShInner::SortedMulti(ref smv) => smv.encode(),
            ShInner::Ms(ref ms) => ms.encode(),
        };
        let address = Address::p2sh(&script, network)?;

        Ok(address)
    }

    /// Obtain the underlying miniscript for this descriptor
    pub fn inner_script(&self) -> ScriptBuf {
        match self.inner {
            ShInner::SortedMulti(ref smv) => smv.encode(),
            ShInner::Ms(ref ms) => ms.encode(),
        }
    }

    /// Obtains the pre bip-340 signature script code for this descriptor.
    pub fn ecdsa_sighash_script_code(&self) -> ScriptBuf {
        match self.inner {
            ShInner::SortedMulti(ref smv) => smv.encode(),
            // For "legacy" P2SH outputs, it is defined as the txo's redeemScript.
            ShInner::Ms(ref ms) => ms.encode(),
        }
    }

    /// Computes the scriptSig that will be in place for an unsigned input
    /// spending an output with this descriptor. For pre-segwit descriptors,
    /// which use the scriptSig for signatures, this returns the empty script.
    ///
    /// This is used in Segwit transactions to produce an unsigned transaction
    /// whose txid will not change during signing (since only the witness data
    /// will change).
    pub fn unsigned_script_sig(&self) -> ScriptBuf {
        match self.inner {
            ShInner::SortedMulti(..) | ShInner::Ms(..) => ScriptBuf::new(),
        }
    }

    /// Returns satisfying non-malleable witness and scriptSig with minimum
    /// weight to spend an output controlled by the given descriptor if it is
    /// possible to construct one using the `satisfier`.
    pub fn get_satisfaction<S>(&self, satisfier: S) -> Result<(Vec<Vec<u8>>, ScriptBuf), Error>
    where
        S: Satisfier<Pk>,
    {
        let script_sig = self.unsigned_script_sig();
        match self.inner {
            ShInner::SortedMulti(ref smv) => {
                let mut script_witness = smv.satisfy(satisfier)?;
                script_witness.push(smv.encode().into_bytes());
                let script_sig = witness_to_scriptsig(&script_witness);
                let witness = vec![];
                Ok((witness, script_sig))
            }
            ShInner::Ms(ref ms) => {
                let mut script_witness = ms.satisfy(satisfier)?;
                script_witness.push(ms.encode().into_bytes());
                let script_sig = witness_to_scriptsig(&script_witness);
                let witness = vec![];
                Ok((witness, script_sig))
            }
        }
    }

    /// Returns satisfying, possibly malleable, witness and scriptSig with
    /// minimum weight to spend an output controlled by the given descriptor if
    /// it is possible to construct one using the `satisfier`.
    pub fn get_satisfaction_mall<S>(&self, satisfier: S) -> Result<(Vec<Vec<u8>>, ScriptBuf), Error>
    where
        S: Satisfier<Pk>,
    {
        let script_sig = self.unsigned_script_sig();
        match self.inner {
            ShInner::Ms(ref ms) => {
                let mut script_witness = ms.satisfy_malleable(satisfier)?;
                script_witness.push(ms.encode().into_bytes());
                let script_sig = witness_to_scriptsig(&script_witness);
                let witness = vec![];
                Ok((witness, script_sig))
            }
            _ => self.get_satisfaction(satisfier),
        }
    }
}

impl Sh<DefiniteDescriptorKey> {
    /// Returns a plan if the provided assets are sufficient to produce a non-malleable satisfaction
    pub fn plan_satisfaction<P>(
        &self,
        provider: &P,
    ) -> Satisfaction<Placeholder<DefiniteDescriptorKey>>
    where
        P: AssetProvider<DefiniteDescriptorKey>,
    {
        match &self.inner {
            ShInner::SortedMulti(ref smv) => smv.build_template(provider),
            ShInner::Ms(ref ms) => ms.build_template(provider),
        }
    }

    /// Returns a plan if the provided assets are sufficient to produce a malleable satisfaction
    pub fn plan_satisfaction_mall<P>(
        &self,
        provider: &P,
    ) -> Satisfaction<Placeholder<DefiniteDescriptorKey>>
    where
        P: AssetProvider<DefiniteDescriptorKey>,
    {
        match &self.inner {
            ShInner::Ms(ref ms) => ms.build_template_mall(provider),
            _ => self.plan_satisfaction(provider),
        }
    }
}

impl<Pk: MiniscriptKey> ForEachKey<Pk> for Sh<Pk> {
    fn for_each_key<'a, F: FnMut(&'a Pk) -> bool>(&'a self, pred: F) -> bool {
        match self.inner {
            ShInner::SortedMulti(ref smv) => smv.for_each_key(pred),
            ShInner::Ms(ref ms) => ms.for_each_key(pred),
        }
    }
}

impl<P, Q> TranslatePk<P, Q> for Sh<P>
where
    P: MiniscriptKey,
    Q: MiniscriptKey,
{
    type Output = Sh<Q>;

    fn translate_pk<T, E>(&self, t: &mut T) -> Result<Self::Output, TranslateErr<E>>
    where
        T: Translator<P, Q, E>,
    {
        let inner = match self.inner {
            ShInner::SortedMulti(ref smv) => ShInner::SortedMulti(smv.translate_pk(t)?),
            ShInner::Ms(ref ms) => ShInner::Ms(ms.translate_pk(t)?),
        };
        Ok(Sh { inner })
    }
}
