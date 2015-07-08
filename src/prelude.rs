//! Reexports of the most common traits, types, and functions.

pub use random::{self, Source};

pub use distribution::{Continuous, Discrete, Distribution};

pub use distribution::Bernoulli;
pub use distribution::Beta;
pub use distribution::Binomial;
pub use distribution::Categorical;
pub use distribution::Exponential;
pub use distribution::Gamma;
pub use distribution::Gaussian;
pub use distribution::Uniform;

pub use sampler::Independent;
