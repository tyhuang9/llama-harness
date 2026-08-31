use crate::{SandboxError, SandboxErrorCode};

/// Immutable library ceilings. Hosts may only make these limits smaller.
pub const HARD_LIMITS: SandboxLimits = SandboxLimits {
    max_program_bytes: 256 * 1024,
    max_ast_nodes: 8_192,
    max_nesting: 64,
    max_bytecode_instructions: 32_768,
    max_constant_bytes: 1024 * 1024,
    max_locals: 512,
    max_operand_stack: 1_024,
    max_control_stack: 64,
    max_fuel: 1_000_000,
    max_slice_fuel: 4_096,
    max_collection_items: 4_096,
    max_loop_iterations: 4_096,
    max_yields: 64,
    max_fanout: 8,
    max_live_bytes: 16 * 1024 * 1024,
    max_cumulative_bytes: 64 * 1024 * 1024,
    max_output_bytes: 1024 * 1024,
};

/// Resource limits applied while parsing, compiling, and executing a program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SandboxLimits {
    /// Maximum raw UTF-8 JSON program bytes.
    pub max_program_bytes: usize,
    /// Maximum total syntax-tree nodes.
    pub max_ast_nodes: usize,
    /// Maximum JSON and language nesting depth.
    pub max_nesting: usize,
    /// Maximum private bytecode instructions.
    pub max_bytecode_instructions: usize,
    /// Maximum serialized literal bytes retained by a program.
    pub max_constant_bytes: usize,
    /// Maximum immutable local bindings.
    pub max_locals: usize,
    /// Maximum VM operand stack entries.
    pub max_operand_stack: usize,
    /// Maximum VM control stack entries.
    pub max_control_stack: usize,
    /// Maximum fuel across one execution.
    pub max_fuel: u64,
    /// Maximum fuel consumed by one scheduling slice.
    pub max_slice_fuel: u64,
    /// Maximum items in any collection.
    pub max_collection_items: usize,
    /// Maximum iterations declared by a loop operation.
    pub max_loop_iterations: usize,
    /// Maximum tool-batch yields.
    pub max_yields: usize,
    /// Maximum calls in one read-only fan-out batch.
    pub max_fanout: usize,
    /// Maximum conservatively retained VM-owned bytes.
    ///
    /// The VM may retain a charge until execution ends even after a scoped
    /// value is no longer reachable. This fail-closed accounting makes the
    /// bound deterministic and can reject earlier than exact heap liveness.
    pub max_live_bytes: usize,
    /// Maximum cumulatively allocated VM-owned bytes.
    pub max_cumulative_bytes: usize,
    /// Maximum serialized return value bytes.
    pub max_output_bytes: usize,
}

impl Default for SandboxLimits {
    fn default() -> Self {
        Self {
            max_program_bytes: 64 * 1024,
            max_ast_nodes: 2_048,
            max_nesting: 32,
            max_bytecode_instructions: 8_192,
            max_constant_bytes: 256 * 1024,
            max_locals: 128,
            max_operand_stack: 256,
            max_control_stack: 32,
            max_fuel: 100_000,
            max_slice_fuel: 1_024,
            max_collection_items: 1_024,
            max_loop_iterations: 1_024,
            max_yields: 16,
            max_fanout: 8,
            max_live_bytes: 4 * 1024 * 1024,
            max_cumulative_bytes: 16 * 1024 * 1024,
            max_output_bytes: 1024 * 1024,
        }
    }
}

impl SandboxLimits {
    /// Validates that every limit is nonzero and no greater than its hard cap.
    pub fn validate(&self) -> Result<(), SandboxError> {
        macro_rules! bounded {
            ($field:ident) => {
                if self.$field == 0 || self.$field > HARD_LIMITS.$field {
                    return Err(SandboxError::new(
                        SandboxErrorCode::InvalidLimits,
                        concat!(
                            stringify!($field),
                            " must be nonzero and within the hard cap"
                        ),
                    ));
                }
            };
        }

        bounded!(max_program_bytes);
        bounded!(max_ast_nodes);
        bounded!(max_nesting);
        bounded!(max_bytecode_instructions);
        bounded!(max_constant_bytes);
        bounded!(max_locals);
        bounded!(max_operand_stack);
        bounded!(max_control_stack);
        bounded!(max_fuel);
        bounded!(max_slice_fuel);
        bounded!(max_collection_items);
        bounded!(max_loop_iterations);
        bounded!(max_yields);
        bounded!(max_fanout);
        bounded!(max_live_bytes);
        bounded!(max_cumulative_bytes);
        bounded!(max_output_bytes);
        if self.max_slice_fuel > self.max_fuel {
            return Err(SandboxError::new(
                SandboxErrorCode::InvalidLimits,
                "max_slice_fuel cannot exceed max_fuel",
            ));
        }
        if self.max_live_bytes > self.max_cumulative_bytes {
            return Err(SandboxError::new(
                SandboxErrorCode::InvalidLimits,
                "max_live_bytes cannot exceed max_cumulative_bytes",
            ));
        }
        Ok(())
    }

    /// Returns the component-wise minimum of host and provider limits.
    pub fn constrained_by(self, other: Self) -> Self {
        Self {
            max_program_bytes: self.max_program_bytes.min(other.max_program_bytes),
            max_ast_nodes: self.max_ast_nodes.min(other.max_ast_nodes),
            max_nesting: self.max_nesting.min(other.max_nesting),
            max_bytecode_instructions: self
                .max_bytecode_instructions
                .min(other.max_bytecode_instructions),
            max_constant_bytes: self.max_constant_bytes.min(other.max_constant_bytes),
            max_locals: self.max_locals.min(other.max_locals),
            max_operand_stack: self.max_operand_stack.min(other.max_operand_stack),
            max_control_stack: self.max_control_stack.min(other.max_control_stack),
            max_fuel: self.max_fuel.min(other.max_fuel),
            max_slice_fuel: self.max_slice_fuel.min(other.max_slice_fuel),
            max_collection_items: self.max_collection_items.min(other.max_collection_items),
            max_loop_iterations: self.max_loop_iterations.min(other.max_loop_iterations),
            max_yields: self.max_yields.min(other.max_yields),
            max_fanout: self.max_fanout.min(other.max_fanout),
            max_live_bytes: self.max_live_bytes.min(other.max_live_bytes),
            max_cumulative_bytes: self.max_cumulative_bytes.min(other.max_cumulative_bytes),
            max_output_bytes: self.max_output_bytes.min(other.max_output_bytes),
        }
    }
}
