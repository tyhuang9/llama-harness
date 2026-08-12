use crate::{HarnessError, RunRequest, ToolCallContext, ToolDefinition};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum PolicyDecision {
    Allow { reason: String },
    Deny { reason: String },
    RequireApproval { reason: String },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ApprovalRecord {
    pub call_id: String,
    pub tool_id: String,
    pub granted: bool,
    pub reason: String,
}

#[async_trait]
pub trait PolicyEngine: Send + Sync {
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
}

#[async_trait]
pub trait ApprovalHandler: Send + Sync {
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
