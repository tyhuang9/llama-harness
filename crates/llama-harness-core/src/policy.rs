use crate::{HarnessError, RunRequest, ToolCallContext, ToolDefinition};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "outcome", rename_all = "snake_case")]
#[non_exhaustive]
/// Policy outcome for a proposed tool call.
pub enum PolicyDecision {
    /// Permit the tool call without approval.
    Allow {
        /// Explanation for allowing the call.
        reason: String,
    },
    /// Reject the tool call.
    Deny {
        /// Explanation for denying the call.
        reason: String,
    },
    /// Pause for an approval decision before execution.
    RequireApproval {
        /// Explanation for requiring approval.
        reason: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
/// Record of an approval decision for a tool call.
pub struct ApprovalRecord {
    /// Identifier of the related tool call.
    pub call_id: String,
    /// Identifier of the related tool.
    pub tool_id: String,
    /// Whether approval was granted.
    pub granted: bool,
    /// Explanation for the approval result.
    pub reason: String,
}

impl ApprovalRecord {
    /// Creates an approval decision for one tool call.
    pub fn new(
        call_id: impl Into<String>,
        tool_id: impl Into<String>,
        granted: bool,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            call_id: call_id.into(),
            tool_id: tool_id.into(),
            granted,
            reason: reason.into(),
        }
    }
}

#[async_trait]
/// Evaluates whether proposed tool calls may proceed.
pub trait PolicyEngine: Send + Sync {
    /// Decides the policy outcome for a proposed tool call.
    async fn decide(
        &self,
        tool: &ToolDefinition,
        arguments: &Value,
        request: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError>;

    /// Decides with immutable correlation for protocol-backed policy handlers.
    async fn decide_with_context(
        &self,
        _: &ToolCallContext,
        tool: &ToolDefinition,
        arguments: &Value,
        request: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        self.decide(tool, arguments, request).await
    }

    /// Independently decides whether a call may cross the speculative issue boundary.
    ///
    /// Ordinary allow decisions and approvals never authorize speculation. Hosts
    /// must explicitly override this method and return [`PolicyDecision::Allow`]
    /// for the exact candidate arguments. The broker re-evaluates this decision
    /// before commit.
    async fn decide_speculative(
        &self,
        _: &ToolCallContext,
        _: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        Ok(PolicyDecision::Deny {
            reason: "speculative execution requires explicit host policy opt-in".into(),
        })
    }
}

#[async_trait]
/// Handles approval requests for policy-gated tool calls.
pub trait ApprovalHandler: Send + Sync {
    /// Decides whether a proposed tool call is approved.
    async fn approve(
        &self,
        tool: &ToolDefinition,
        arguments: &Value,
        request: &RunRequest,
    ) -> Result<ApprovalRecord, HarnessError>;

    /// Approves with immutable correlation for protocol-backed approval handlers.
    async fn approve_with_context(
        &self,
        _: &ToolCallContext,
        tool: &ToolDefinition,
        arguments: &Value,
        request: &RunRequest,
    ) -> Result<ApprovalRecord, HarnessError> {
        self.approve(tool, arguments, request).await
    }
}

/// Policy implementation that allows every tool call.
pub struct AllowAllPolicy;

#[async_trait]
impl PolicyEngine for AllowAllPolicy {
    async fn decide(
        &self,
        _: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        Ok(PolicyDecision::Allow {
            reason: "explicit allow-all policy".into(),
        })
    }
}

/// Default policy: reads are allowed; tools that can change state are denied.
pub struct SafeDefaultPolicy;

#[async_trait]
impl PolicyEngine for SafeDefaultPolicy {
    async fn decide(
        &self,
        tool: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        if tool.read_only {
            Ok(PolicyDecision::Allow {
                reason: "read-only tool allowed by default policy".into(),
            })
        } else {
            Ok(PolicyDecision::Deny {
                reason: "state-changing tool requires an explicit policy".into(),
            })
        }
    }
}

/// Approval implementation that denies every approval request.
pub struct DenyApproval;

#[async_trait]
impl ApprovalHandler for DenyApproval {
    async fn approve(
        &self,
        tool: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<ApprovalRecord, HarnessError> {
        Ok(ApprovalRecord {
            call_id: String::new(),
            tool_id: tool.id.clone(),
            granted: false,
            reason: "no approval handler configured".into(),
        })
    }
}
