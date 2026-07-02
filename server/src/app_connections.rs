use anyhow::{Context, Result};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use ring::digest::{digest, SHA256};
use serde::{Deserialize, Serialize};
use std::path::Path;
use subtle::ConstantTimeEq;
use tokio::fs;
use uuid::Uuid;

pub const DEFAULT_APP_SCOPES: [&str; 4] = [
    "capabilities:read",
    "runs:create",
    "runs:stream",
    "tool-results:submit",
];
pub const PAIRING_TTL_MINUTES: i64 = 10;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AppConnectionStore {
    pub pairings: Vec<AppPairing>,
    pub tokens: Vec<AppTokenRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppPairing {
    pub id: String,
    pub app_id: String,
    pub app_name: String,
    pub requested_scopes: Vec<String>,
    pub origin: Option<String>,
    pub redirect_uri: Option<String>,
    pub user_code: String,
    pub secret_hash: String,
    pub status: PairingStatus,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub approved_at: Option<DateTime<Utc>>,
    pub denied_at: Option<DateTime<Utc>>,
    pub delivered_at: Option<DateTime<Utc>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PairingStatus {
    Pending,
    Approved,
    Denied,
    Expired,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppTokenRecord {
    pub id: String,
    pub app_id: String,
    pub name: String,
    pub token_hash: String,
    pub scopes: Vec<String>,
    pub origin: Option<String>,
    pub kind: AppTokenKind,
    pub created_at: DateTime<Utc>,
    pub last_used_at: Option<DateTime<Utc>>,
    pub expires_at: Option<DateTime<Utc>>,
    pub revoked_at: Option<DateTime<Utc>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AppTokenKind {
    Pairing,
    Service,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AppTokenSummary {
    pub id: String,
    pub app_id: String,
    pub name: String,
    pub scopes: Vec<String>,
    pub origin: Option<String>,
    pub kind: AppTokenKind,
    pub created_at: DateTime<Utc>,
    pub last_used_at: Option<DateTime<Utc>>,
    pub expires_at: Option<DateTime<Utc>>,
    pub revoked_at: Option<DateTime<Utc>>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AppPairingSummary {
    pub id: String,
    pub app_id: String,
    pub app_name: String,
    pub requested_scopes: Vec<String>,
    pub origin: Option<String>,
    pub redirect_uri: Option<String>,
    pub user_code: String,
    pub status: PairingStatus,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub approved_at: Option<DateTime<Utc>>,
    pub denied_at: Option<DateTime<Utc>>,
    pub delivered_at: Option<DateTime<Utc>>,
}

#[derive(Clone, Debug)]
pub struct StartedPairing {
    pub pairing: AppPairing,
    pub pairing_secret: String,
}

#[derive(Clone, Debug)]
pub struct IssuedToken {
    pub record: AppTokenRecord,
    pub token: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AppAuthError {
    MissingToken,
    InvalidToken,
    RevokedToken,
    ExpiredToken,
    WrongApp,
    MissingScope,
    PairingNotFound,
    PairingExpired,
    PairingDenied,
    PairingPending,
    PairingAlreadyDelivered,
}

impl AppAuthError {
    pub fn message(&self) -> &'static str {
        match self {
            AppAuthError::MissingToken => "missing bearer token",
            AppAuthError::InvalidToken => "invalid bearer token",
            AppAuthError::RevokedToken => "token has been revoked",
            AppAuthError::ExpiredToken => "token has expired",
            AppAuthError::WrongApp => "token is not valid for this app",
            AppAuthError::MissingScope => "token is missing the required scope",
            AppAuthError::PairingNotFound => "pairing request was not found",
            AppAuthError::PairingExpired => "pairing request has expired",
            AppAuthError::PairingDenied => "pairing request was denied",
            AppAuthError::PairingPending => "pairing request is still pending",
            AppAuthError::PairingAlreadyDelivered => "pairing token was already delivered",
        }
    }
}

impl From<&AppTokenRecord> for AppTokenSummary {
    fn from(record: &AppTokenRecord) -> Self {
        Self {
            id: record.id.clone(),
            app_id: record.app_id.clone(),
            name: record.name.clone(),
            scopes: record.scopes.clone(),
            origin: record.origin.clone(),
            kind: record.kind.clone(),
            created_at: record.created_at,
            last_used_at: record.last_used_at,
            expires_at: record.expires_at,
            revoked_at: record.revoked_at,
        }
    }
}

impl From<&AppPairing> for AppPairingSummary {
    fn from(pairing: &AppPairing) -> Self {
        Self {
            id: pairing.id.clone(),
            app_id: pairing.app_id.clone(),
            app_name: pairing.app_name.clone(),
            requested_scopes: pairing.requested_scopes.clone(),
            origin: pairing.origin.clone(),
            redirect_uri: pairing.redirect_uri.clone(),
            user_code: pairing.user_code.clone(),
            status: pairing.status.clone(),
            created_at: pairing.created_at,
            expires_at: pairing.expires_at,
            approved_at: pairing.approved_at,
            denied_at: pairing.denied_at,
            delivered_at: pairing.delivered_at,
        }
    }
}

impl AppConnectionStore {
    pub fn start_pairing(
        &mut self,
        app_id: String,
        app_name: String,
        requested_scopes: Vec<String>,
        origin: Option<String>,
        redirect_uri: Option<String>,
        now: DateTime<Utc>,
    ) -> StartedPairing {
        self.expire_stale_pairings(now);
        let pairing_secret = generate_pairing_secret();
        let pairing = AppPairing {
            id: Uuid::new_v4().to_string(),
            app_id,
            app_name,
            requested_scopes: normalize_scopes(requested_scopes),
            origin: clean_optional(origin),
            redirect_uri: clean_optional(redirect_uri),
            user_code: user_code(),
            secret_hash: hash_secret(&pairing_secret),
            status: PairingStatus::Pending,
            created_at: now,
            expires_at: now + ChronoDuration::minutes(PAIRING_TTL_MINUTES),
            approved_at: None,
            denied_at: None,
            delivered_at: None,
        };
        self.pairings.push(pairing.clone());
        StartedPairing {
            pairing,
            pairing_secret,
        }
    }

    pub fn approve_pairing(
        &mut self,
        pairing_id: &str,
        approved_scopes: Option<Vec<String>>,
        now: DateTime<Utc>,
    ) -> Result<AppPairingSummary, AppAuthError> {
        let pairing = self
            .pairing_mut(pairing_id)
            .ok_or(AppAuthError::PairingNotFound)?;
        expire_pairing_if_needed(pairing, now);
        match pairing.status {
            PairingStatus::Expired => return Err(AppAuthError::PairingExpired),
            PairingStatus::Denied => return Err(AppAuthError::PairingDenied),
            PairingStatus::Approved => return Ok(AppPairingSummary::from(&*pairing)),
            PairingStatus::Pending => {}
        }

        if let Some(scopes) = approved_scopes {
            pairing.requested_scopes = normalize_scopes(scopes);
        }
        pairing.status = PairingStatus::Approved;
        pairing.approved_at = Some(now);
        Ok(AppPairingSummary::from(&*pairing))
    }

    pub fn deny_pairing(
        &mut self,
        pairing_id: &str,
        now: DateTime<Utc>,
    ) -> Result<AppPairingSummary, AppAuthError> {
        let pairing = self
            .pairing_mut(pairing_id)
            .ok_or(AppAuthError::PairingNotFound)?;
        expire_pairing_if_needed(pairing, now);
        if pairing.status == PairingStatus::Expired {
            return Err(AppAuthError::PairingExpired);
        }
        pairing.status = PairingStatus::Denied;
        pairing.denied_at = Some(now);
        Ok(AppPairingSummary::from(&*pairing))
    }

    pub fn exchange_pairing(
        &mut self,
        pairing_id: &str,
        pairing_secret: &str,
        now: DateTime<Utc>,
    ) -> Result<IssuedToken, AppAuthError> {
        let pairing = self
            .pairing_mut(pairing_id)
            .ok_or(AppAuthError::PairingNotFound)?;
        expire_pairing_if_needed(pairing, now);
        if !verify_secret(pairing_secret, &pairing.secret_hash) {
            return Err(AppAuthError::InvalidToken);
        }
        match pairing.status {
            PairingStatus::Pending => return Err(AppAuthError::PairingPending),
            PairingStatus::Denied => return Err(AppAuthError::PairingDenied),
            PairingStatus::Expired => return Err(AppAuthError::PairingExpired),
            PairingStatus::Approved => {}
        }
        if pairing.delivered_at.is_some() {
            return Err(AppAuthError::PairingAlreadyDelivered);
        }

        let raw_token = generate_app_token();
        let token_record = AppTokenRecord {
            id: Uuid::new_v4().to_string(),
            app_id: pairing.app_id.clone(),
            name: format!("{} pairing", pairing.app_name),
            token_hash: hash_secret(&raw_token),
            scopes: pairing.requested_scopes.clone(),
            origin: pairing.origin.clone(),
            kind: AppTokenKind::Pairing,
            created_at: now,
            last_used_at: None,
            expires_at: None,
            revoked_at: None,
        };
        pairing.delivered_at = Some(now);
        self.tokens.push(token_record.clone());
        Ok(IssuedToken {
            record: token_record,
            token: raw_token,
        })
    }

    pub fn issue_service_token(
        &mut self,
        app_id: String,
        name: String,
        scopes: Vec<String>,
        origin: Option<String>,
        expires_at: Option<DateTime<Utc>>,
        now: DateTime<Utc>,
    ) -> IssuedToken {
        let raw_token = generate_app_token();
        let record = AppTokenRecord {
            id: Uuid::new_v4().to_string(),
            app_id,
            name,
            token_hash: hash_secret(&raw_token),
            scopes: normalize_scopes(scopes),
            origin: clean_optional(origin),
            kind: AppTokenKind::Service,
            created_at: now,
            last_used_at: None,
            expires_at,
            revoked_at: None,
        };
        self.tokens.push(record.clone());
        IssuedToken {
            record,
            token: raw_token,
        }
    }

    pub fn revoke_token(
        &mut self,
        app_id: &str,
        token_id: &str,
        now: DateTime<Utc>,
    ) -> Result<AppTokenSummary, AppAuthError> {
        let token = self
            .tokens
            .iter_mut()
            .find(|token| token.id == token_id && token.app_id.eq_ignore_ascii_case(app_id))
            .ok_or(AppAuthError::InvalidToken)?;
        token.revoked_at = Some(now);
        Ok(AppTokenSummary::from(&*token))
    }

    pub fn authorize(
        &mut self,
        raw_token: &str,
        app_id: &str,
        required_scope: &str,
        now: DateTime<Utc>,
    ) -> Result<(), AppAuthError> {
        let token = self
            .tokens
            .iter_mut()
            .find(|token| verify_secret(raw_token, &token.token_hash))
            .ok_or(AppAuthError::InvalidToken)?;
        if token.revoked_at.is_some() {
            return Err(AppAuthError::RevokedToken);
        }
        if token
            .expires_at
            .map(|expires_at| expires_at <= now)
            .unwrap_or(false)
        {
            return Err(AppAuthError::ExpiredToken);
        }
        if !token.app_id.eq_ignore_ascii_case(app_id.trim()) {
            return Err(AppAuthError::WrongApp);
        }
        if !has_scope(&token.scopes, required_scope) {
            return Err(AppAuthError::MissingScope);
        }
        token.last_used_at = Some(now);
        Ok(())
    }

    pub fn pairings(&self) -> Vec<AppPairingSummary> {
        self.pairings.iter().map(AppPairingSummary::from).collect()
    }

    pub fn tokens(&self) -> Vec<AppTokenSummary> {
        self.tokens.iter().map(AppTokenSummary::from).collect()
    }

    pub fn expire_stale_pairings(&mut self, now: DateTime<Utc>) {
        for pairing in &mut self.pairings {
            expire_pairing_if_needed(pairing, now);
        }
    }

    fn pairing_mut(&mut self, pairing_id: &str) -> Option<&mut AppPairing> {
        self.pairings
            .iter_mut()
            .find(|pairing| pairing.id == pairing_id)
    }
}

pub async fn load_connections(path: &Path) -> Result<AppConnectionStore> {
    match fs::read_to_string(path).await {
        Ok(contents) => serde_json::from_str(&contents)
            .with_context(|| format!("failed to parse app connections at {}", path.display())),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(AppConnectionStore::default()),
        Err(err) => Err(err).with_context(|| format!("failed to read {}", path.display())),
    }
}

pub async fn save_connections(path: &Path, store: &AppConnectionStore) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .await
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }

    let contents = serde_json::to_string_pretty(store)?;
    fs::write(path, format!("{contents}\n"))
        .await
        .with_context(|| format!("failed to write {}", path.display()))
}

pub fn normalize_scopes(scopes: Vec<String>) -> Vec<String> {
    let mut normalized = scopes
        .into_iter()
        .map(|scope| scope.trim().to_ascii_lowercase())
        .filter(|scope| !scope.is_empty())
        .collect::<Vec<_>>();
    if normalized.is_empty() {
        normalized = DEFAULT_APP_SCOPES
            .iter()
            .map(|scope| scope.to_string())
            .collect();
    }
    normalized.sort();
    normalized.dedup();
    normalized
}

pub fn bearer_token(header: Option<&str>) -> Result<&str, AppAuthError> {
    let Some(header) = header else {
        return Err(AppAuthError::MissingToken);
    };
    let Some(token) = header.trim().strip_prefix("Bearer ") else {
        return Err(AppAuthError::MissingToken);
    };
    let token = token.trim();
    if token.is_empty() {
        return Err(AppAuthError::MissingToken);
    }
    Ok(token)
}

fn generate_app_token() -> String {
    format!(
        "lh_app_{}_{}",
        Uuid::new_v4().simple(),
        Uuid::new_v4().simple()
    )
}

fn generate_pairing_secret() -> String {
    format!(
        "lh_pair_{}_{}",
        Uuid::new_v4().simple(),
        Uuid::new_v4().simple()
    )
}

fn user_code() -> String {
    let raw = Uuid::new_v4().simple().to_string();
    format!("{}-{}", &raw[0..4], &raw[4..8]).to_ascii_uppercase()
}

fn hash_secret(secret: &str) -> String {
    let hash = digest(&SHA256, secret.as_bytes());
    format!("sha256:{}", hex(hash.as_ref()))
}

fn verify_secret(secret: &str, stored_hash: &str) -> bool {
    let expected = hash_secret(secret);
    let expected = expected.as_bytes();
    let stored = stored_hash.as_bytes();
    expected.len() == stored.len() && expected.ct_eq(stored).into()
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn has_scope(scopes: &[String], required_scope: &str) -> bool {
    scopes
        .iter()
        .any(|scope| scope == "*" || scope.eq_ignore_ascii_case(required_scope))
}

fn clean_optional(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn expire_pairing_if_needed(pairing: &mut AppPairing, now: DateTime<Utc>) {
    if pairing.status == PairingStatus::Pending && pairing.expires_at <= now {
        pairing.status = PairingStatus::Expired;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairing_exchange_delivers_token_once() {
        let now = Utc::now();
        let mut store = AppConnectionStore::default();
        let started = store.start_pairing(
            "note".to_string(),
            "Note".to_string(),
            vec!["runs:create".to_string()],
            None,
            None,
            now,
        );

        store
            .approve_pairing(&started.pairing.id, None, now)
            .expect("pairing should approve");
        let issued = store
            .exchange_pairing(&started.pairing.id, &started.pairing_secret, now)
            .expect("approved pairing should exchange");

        assert!(issued.token.starts_with("lh_app_"));
        assert_eq!(store.tokens.len(), 1);
        assert_eq!(
            store
                .exchange_pairing(&started.pairing.id, &started.pairing_secret, now)
                .expect_err("token should only be delivered once"),
            AppAuthError::PairingAlreadyDelivered
        );
    }

    #[test]
    fn app_authorization_requires_matching_app_and_scope() {
        let now = Utc::now();
        let mut store = AppConnectionStore::default();
        let issued = store.issue_service_token(
            "note".to_string(),
            "Note service".to_string(),
            vec!["runs:create".to_string()],
            None,
            None,
            now,
        );

        assert!(store
            .authorize(&issued.token, "note", "runs:create", now)
            .is_ok());
        assert_eq!(
            store
                .authorize(&issued.token, "other", "runs:create", now)
                .expect_err("wrong app should be rejected"),
            AppAuthError::WrongApp
        );
        assert_eq!(
            store
                .authorize(&issued.token, "note", "tool-results:submit", now)
                .expect_err("missing scope should be rejected"),
            AppAuthError::MissingScope
        );
    }

    #[test]
    fn revoked_token_is_rejected() {
        let now = Utc::now();
        let mut store = AppConnectionStore::default();
        let issued = store.issue_service_token(
            "note".to_string(),
            "Note service".to_string(),
            DEFAULT_APP_SCOPES
                .iter()
                .map(|scope| scope.to_string())
                .collect(),
            None,
            None,
            now,
        );

        store
            .revoke_token("note", &issued.record.id, now)
            .expect("token should revoke");
        assert_eq!(
            store
                .authorize(&issued.token, "note", "runs:create", now)
                .expect_err("revoked token should fail"),
            AppAuthError::RevokedToken
        );
    }
}
