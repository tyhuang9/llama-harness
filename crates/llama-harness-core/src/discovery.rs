use crate::{
    runner::check_stopped, HarnessError, ProviderCapabilityLimits, ToolCaller, ToolDefinition,
    ToolRegistry,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::Arc,
};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

/// Version of the canonical safe-metadata catalog fingerprint format.
pub const CATALOG_FINGERPRINT_VERSION: u32 = 1;
const MAX_DISCOVERY_QUERY_BYTES: usize = 4096;
const MAX_QUERY_TERMS: usize = 64;
pub(crate) const MAX_DISCOVERY_IDENTIFIER_BYTES: usize = 128;
pub(crate) const DISCOVERY_GUARD_INTERVAL: usize = 64;

/// Whether a registered tool is always exposed or selected only when relevant.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolExposure {
    /// The tool is included in every non-empty model tool scope.
    #[default]
    Hot,
    /// The tool is included only when the complete catalog does not fit and discovery selects it.
    Deferred,
}

/// Safe, indexable discovery metadata associated with a registered tool.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
#[non_exhaustive]
pub struct ToolDiscoveryMetadata {
    /// Whether the tool is hot or deferred.
    pub exposure: ToolExposure,
    /// Optional stable namespace used for exact and lexical selection.
    pub namespace: Option<String>,
    /// Stable alternate identifiers used for discovery.
    pub aliases: Vec<String>,
}

impl ToolDiscoveryMetadata {
    /// Creates conservative metadata that keeps the tool hot.
    pub fn hot() -> Self {
        Self::default()
    }

    /// Creates deferred metadata with no namespace or aliases.
    pub fn deferred() -> Self {
        Self {
            exposure: ToolExposure::Deferred,
            ..Self::default()
        }
    }

    /// Sets the stable namespace.
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    /// Replaces the stable aliases.
    pub fn with_aliases(mut self, aliases: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.aliases = aliases.into_iter().map(Into::into).collect();
        self
    }

    pub(crate) fn validate(&self, tool_id: &str) -> Result<(), HarnessError> {
        if let Some(namespace) = &self.namespace {
            validate_stable_identifier("namespace", namespace, tool_id)?;
        }
        if self.aliases.len() > 32 {
            return Err(HarnessError::InvalidTool(format!(
                "tool {tool_id} has more than 32 discovery aliases"
            )));
        }
        let mut unique = BTreeSet::new();
        for alias in &self.aliases {
            validate_stable_identifier("alias", alias, tool_id)?;
            let normalized = tokenize(alias).join(" ");
            if !unique.insert(normalized) {
                return Err(HarnessError::InvalidTool(format!(
                    "tool {tool_id} has duplicate normalized discovery aliases"
                )));
            }
        }
        Ok(())
    }
}

/// Host limits applied before provider-specific discovery limits.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
#[non_exhaustive]
pub struct ToolDiscoveryLimits {
    /// Maximum tools in one selected scope.
    pub max_tools: u32,
    /// Maximum exact serialized bytes for the selected tool-definition array.
    pub max_tool_schema_bytes: u64,
    /// Maximum deferred candidates admitted for a low-margin lexical match.
    pub max_expansion_tools: u32,
}

impl Default for ToolDiscoveryLimits {
    fn default() -> Self {
        Self {
            max_tools: 64,
            max_tool_schema_bytes: 128 * 1024,
            max_expansion_tools: 8,
        }
    }
}

impl ToolDiscoveryLimits {
    /// Creates the default bounded host limits.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum selected tool count.
    pub fn with_max_tools(mut self, max_tools: u32) -> Self {
        self.max_tools = max_tools;
        self
    }

    /// Sets the maximum exact serialized schema bytes.
    pub fn with_max_tool_schema_bytes(mut self, bytes: u64) -> Self {
        self.max_tool_schema_bytes = bytes;
        self
    }

    /// Sets the bounded low-confidence expansion count.
    pub fn with_max_expansion_tools(mut self, count: u32) -> Self {
        self.max_expansion_tools = count;
        self
    }

    pub(crate) fn effective(self, provider: &ProviderCapabilityLimits) -> EffectiveLimits {
        EffectiveLimits {
            max_tools: provider
                .max_tools
                .map_or(self.max_tools, |value| value.min(self.max_tools)),
            max_bytes: provider
                .max_tool_schema_bytes
                .map_or(self.max_tool_schema_bytes, |value| {
                    value.min(self.max_tool_schema_bytes)
                }),
            max_expansion: self.max_expansion_tools,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct EffectiveLimits {
    max_tools: u32,
    max_bytes: u64,
    max_expansion: u32,
}

/// Versioned BLAKE3 identity for an immutable safe-metadata catalog index.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct CatalogFingerprint {
    /// Canonical fingerprint format version.
    pub version: u32,
    /// Lowercase hexadecimal BLAKE3 digest.
    pub digest: String,
}

/// Immutable set of tools selected for one caller and one run.
#[derive(Clone, Debug)]
pub(crate) struct ToolScope {
    caller: ToolCaller,
    tool_ids: BTreeSet<String>,
    definitions: Arc<[ToolDefinition]>,
}

impl ToolScope {
    pub(crate) fn caller(&self) -> ToolCaller {
        self.caller
    }

    pub(crate) fn contains(&self, tool_id: &str) -> bool {
        self.tool_ids.contains(tool_id)
    }

    pub(crate) fn definitions(&self) -> &[ToolDefinition] {
        &self.definitions
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.definitions.is_empty()
    }

    fn empty(caller: ToolCaller) -> Self {
        Self {
            caller,
            tool_ids: BTreeSet::new(),
            definitions: Arc::from([]),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ToolDiscoveryStats {
    pub(crate) candidate_count: u32,
    pub(crate) selected_count: u32,
    pub(crate) deferred_candidate_count: u32,
    pub(crate) catalog_exceeded_budget: bool,
    pub(crate) cache_hit: bool,
}

#[derive(Clone)]
pub(crate) struct CatalogEntry {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) metadata: ToolDiscoveryMetadata,
    pub(crate) terms: BTreeMap<String, u32>,
}

pub(crate) struct AllowedCatalogEntry<'a> {
    pub(crate) definition: &'a ToolDefinition,
    pub(crate) metadata: &'a ToolDiscoveryMetadata,
    pub(crate) serialized_definition: &'a Arc<[u8]>,
}

pub(crate) struct CatalogIndex {
    entries: Vec<CatalogEntry>,
    positions: HashMap<String, usize>,
    fingerprint: CatalogFingerprint,
}

impl CatalogIndex {
    pub(crate) fn build(
        mut entries: Vec<CatalogEntry>,
        guard: &mut impl FnMut() -> Result<(), HarnessError>,
    ) -> Result<Self, HarnessError> {
        guard()?;
        entries.sort_by(|left, right| left.id.cmp(&right.id));
        for (position, entry) in entries.iter_mut().enumerate() {
            if position % DISCOVERY_GUARD_INTERVAL == 0 {
                guard()?;
            }
            entry.terms = indexed_terms(entry);
        }
        let mut positions = HashMap::with_capacity(entries.len());
        for (position, entry) in entries.iter().enumerate() {
            if position % DISCOVERY_GUARD_INTERVAL == 0 {
                guard()?;
            }
            positions.insert(entry.id.clone(), position);
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"llama-harness-tool-catalog\0v1\0");
        for (position, entry) in entries.iter().enumerate() {
            if position % DISCOVERY_GUARD_INTERVAL == 0 {
                guard()?;
            }
            fingerprint_field(&mut hasher, &entry.id);
            fingerprint_field(&mut hasher, &entry.name);
            fingerprint_field(
                &mut hasher,
                entry.metadata.namespace.as_deref().unwrap_or_default(),
            );
            hasher.update(&[match entry.metadata.exposure {
                ToolExposure::Hot => 0,
                ToolExposure::Deferred => 1,
            }]);
            for alias in &entry.metadata.aliases {
                fingerprint_field(&mut hasher, alias);
            }
            hasher.update(&[0xff]);
        }
        Ok(Self {
            entries,
            positions,
            fingerprint: CatalogFingerprint {
                version: CATALOG_FINGERPRINT_VERSION,
                digest: hasher.finalize().to_hex().to_string(),
            },
        })
    }

    pub(crate) fn fingerprint(&self) -> CatalogFingerprint {
        self.fingerprint.clone()
    }

    fn entry(&self, tool_id: &str) -> Option<&CatalogEntry> {
        self.positions
            .get(tool_id)
            .and_then(|position| self.entries.get(*position))
    }
}

impl ToolRegistry {
    /// Returns the versioned fingerprint for the cached safe-metadata catalog.
    pub fn catalog_fingerprint(&self) -> CatalogFingerprint {
        self.catalog_index().0.fingerprint()
    }

    #[cfg(test)]
    pub(crate) fn select_scope(
        &self,
        query: &str,
        allowlist: &[String],
        caller: ToolCaller,
        host_limits: ToolDiscoveryLimits,
        provider_limits: &ProviderCapabilityLimits,
    ) -> Result<(ToolScope, ToolDiscoveryStats), HarnessError> {
        self.select_scope_guarded(
            query,
            allowlist,
            caller,
            host_limits,
            provider_limits,
            &mut || Ok(()),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn select_scope_for_run(
        &self,
        query: &str,
        allowlist: &[String],
        caller: ToolCaller,
        host_limits: ToolDiscoveryLimits,
        provider_limits: &ProviderCapabilityLimits,
        cancellation: &CancellationToken,
        deadline: Option<Instant>,
    ) -> Result<(ToolScope, ToolDiscoveryStats), HarnessError> {
        self.select_scope_guarded(
            query,
            allowlist,
            caller,
            host_limits,
            provider_limits,
            &mut || check_stopped(cancellation, deadline, "run deadline reached"),
        )
    }

    fn select_scope_guarded(
        &self,
        query: &str,
        allowlist: &[String],
        caller: ToolCaller,
        host_limits: ToolDiscoveryLimits,
        provider_limits: &ProviderCapabilityLimits,
        guard: &mut impl FnMut() -> Result<(), HarnessError>,
    ) -> Result<(ToolScope, ToolDiscoveryStats), HarnessError> {
        guard()?;
        let limits = host_limits.effective(provider_limits);
        let allowed = self.allowed_catalog_guarded(allowlist, caller, guard)?;
        let deferred_count = allowed
            .iter()
            .filter(|entry| entry.metadata.exposure == ToolExposure::Deferred)
            .count();
        let mut base_stats = ToolDiscoveryStats {
            candidate_count: allowed.len() as u32,
            selected_count: 0,
            deferred_candidate_count: deferred_count as u32,
            catalog_exceeded_budget: false,
            cache_hit: false,
        };

        // A serialized tool-definition array is at least `[]`. Treat providers
        // that cannot carry even that representation as having no tool
        // capacity, without constructing or consulting the catalog index.
        if limits.max_tools == 0 || limits.max_bytes < 2 || allowed.is_empty() {
            return Ok((ToolScope::empty(caller), base_stats));
        }
        let all_definitions = allowed
            .iter()
            .map(|entry| entry.definition.clone())
            .collect::<Vec<_>>();
        if fits(&allowed, limits)? {
            return Ok(scope_with_stats(caller, all_definitions, base_stats, false));
        }

        let (index, cache_hit) = self.catalog_index_guarded(guard)?;
        base_stats.cache_hit = cache_hit;

        let mut scope_entries = Vec::with_capacity(allowed.len());
        for (position, entry) in allowed.iter().enumerate() {
            if position % DISCOVERY_GUARD_INTERVAL == 0 {
                guard()?;
            }
            if let Some(indexed) = index.entry(&entry.definition.id) {
                scope_entries.push(indexed);
            }
        }
        scope_entries.sort_by(|left, right| left.id.cmp(&right.id));
        let mut required = allowed
            .iter()
            .filter(|entry| entry.metadata.exposure == ToolExposure::Hot)
            .map(|entry| entry.definition.id.clone())
            .collect::<BTreeSet<_>>();
        let ranked = rank(&scope_entries, query, limits.max_expansion as usize, guard)?;
        let exact = matches!(ranked, RankedSelection::Exact(_));
        let hot = allowed
            .iter()
            .filter(|entry| required.contains(&entry.definition.id))
            .collect::<Vec<_>>();
        if !fits_refs(&hot, limits)? {
            return Err(HarnessError::ResourceLimit(
                "mandatory hot tool scope exceeds discovery budget".into(),
            ));
        }
        if exact {
            required.extend(ranked.ids().iter().cloned());
            let exact_entries = allowed
                .iter()
                .filter(|entry| required.contains(&entry.definition.id))
                .collect::<Vec<_>>();
            if !fits_refs(&exact_entries, limits)? {
                return Err(HarnessError::ResourceLimit(
                    "mandatory hot or exact-match tool scope exceeds discovery budget".into(),
                ));
            }
        } else {
            for id in ranked.ids() {
                let Some(entry) = allowed.iter().find(|entry| entry.definition.id == *id) else {
                    continue;
                };
                let mut expanded = required.clone();
                expanded.insert(entry.definition.id.clone());
                let expanded_entries = allowed
                    .iter()
                    .filter(|entry| expanded.contains(&entry.definition.id))
                    .collect::<Vec<_>>();
                if fits_refs(&expanded_entries, limits)? {
                    required = expanded;
                }
            }
        }
        let selected = allowed
            .iter()
            .filter(|entry| required.contains(&entry.definition.id))
            .map(|entry| entry.definition.clone())
            .collect::<Vec<_>>();
        Ok(scope_with_stats(caller, selected, base_stats, true))
    }
}

fn scope_with_stats(
    caller: ToolCaller,
    definitions: Vec<ToolDefinition>,
    mut stats: ToolDiscoveryStats,
    exceeded: bool,
) -> (ToolScope, ToolDiscoveryStats) {
    stats.selected_count = definitions.len() as u32;
    stats.catalog_exceeded_budget = exceeded;
    let tool_ids = definitions.iter().map(|tool| tool.id.clone()).collect();
    (
        ToolScope {
            caller,
            tool_ids,
            definitions: Arc::from(definitions),
        },
        stats,
    )
}

fn fits(
    entries: &[AllowedCatalogEntry<'_>],
    limits: EffectiveLimits,
) -> Result<bool, HarnessError> {
    fits_lengths(
        entries.len(),
        entries
            .iter()
            .map(|entry| entry.serialized_definition.len() as u64),
        limits,
    )
}

fn fits_refs(
    entries: &[&AllowedCatalogEntry<'_>],
    limits: EffectiveLimits,
) -> Result<bool, HarnessError> {
    fits_lengths(
        entries.len(),
        entries
            .iter()
            .map(|entry| entry.serialized_definition.len() as u64),
        limits,
    )
}

fn fits_lengths(
    count: usize,
    mut lengths: impl Iterator<Item = u64>,
    limits: EffectiveLimits,
) -> Result<bool, HarnessError> {
    let elements = lengths.try_fold(0u64, |total, length| total.checked_add(length));
    let commas = u64::try_from(count.saturating_sub(1)).map_err(|_| {
        HarnessError::ResourceLimit("tool catalog serialized byte accounting overflowed".into())
    })?;
    let bytes = elements
        .and_then(|total| total.checked_add(commas))
        .and_then(|total| total.checked_add(2))
        .ok_or_else(|| {
            HarnessError::ResourceLimit("tool catalog serialized byte accounting overflowed".into())
        })?;
    Ok(count <= limits.max_tools as usize && bytes <= limits.max_bytes)
}

enum RankedSelection {
    Exact(Vec<String>),
    Lexical(Vec<String>),
}

impl RankedSelection {
    fn ids(&self) -> &[String] {
        match self {
            Self::Exact(ids) | Self::Lexical(ids) => ids,
        }
    }
}

fn rank(
    entries: &[&CatalogEntry],
    query: &str,
    max_expansion: usize,
    guard: &mut impl FnMut() -> Result<(), HarnessError>,
) -> Result<RankedSelection, HarnessError> {
    let normalized = normalize_phrase(query);
    let query_terms = tokenize(query);
    guard()?;
    let exact_id = entries
        .iter()
        .filter(|entry| entry.id == normalized)
        .map(|entry| entry.id.clone())
        .collect::<Vec<_>>();
    if exact_id.len() == 1 {
        return Ok(RankedSelection::Exact(exact_id));
    }
    guard()?;
    let exact_name = entries
        .iter()
        .filter(|entry| normalize_phrase(&entry.name) == normalized)
        .map(|entry| entry.id.clone())
        .collect::<Vec<_>>();
    if exact_name.len() == 1 {
        return Ok(RankedSelection::Exact(exact_name));
    }
    if exact_name.len() > 1 {
        return Ok(RankedSelection::Exact(Vec::new()));
    }
    guard()?;
    let exact_namespace = entries
        .iter()
        .filter(|entry| entry.metadata.namespace.as_deref() == Some(normalized.as_str()))
        .map(|entry| entry.id.clone())
        .collect::<Vec<_>>();
    if !exact_namespace.is_empty() {
        return Ok(RankedSelection::Exact(exact_namespace));
    }
    guard()?;
    let exact_alias = entries
        .iter()
        .filter(|entry| {
            entry
                .metadata
                .aliases
                .iter()
                .any(|alias| alias == &normalized)
        })
        .map(|entry| entry.id.clone())
        .collect::<Vec<_>>();
    if exact_alias.len() == 1 {
        return Ok(RankedSelection::Exact(exact_alias));
    }
    if exact_alias.len() > 1 {
        return Ok(RankedSelection::Exact(Vec::new()));
    }
    if max_expansion == 0 {
        return Ok(RankedSelection::Lexical(Vec::new()));
    }
    if query_terms.is_empty() {
        return Ok(RankedSelection::Lexical(Vec::new()));
    }
    let documents = entries.len() as u64;
    let mut document_frequency = BTreeMap::<&str, u64>::new();
    for (position, entry) in entries.iter().enumerate() {
        if position % DISCOVERY_GUARD_INTERVAL == 0 {
            guard()?;
        }
        for term in entry.terms.keys() {
            *document_frequency.entry(term).or_default() += 1;
        }
    }
    let mut scored = Vec::new();
    for (position, entry) in entries.iter().enumerate() {
        if position % DISCOVERY_GUARD_INTERVAL == 0 {
            guard()?;
        }
        let Some(score) = query_terms.iter().try_fold(0u64, |total, term| {
            let Some(frequency) = entry.terms.get(term).copied().map(u64::from) else {
                return Some(total);
            };
            let document_frequency = *document_frequency.get(term.as_str())?;
            let idf = documents
                .checked_add(1)?
                .checked_mul(1_000)?
                .checked_div(document_frequency.checked_add(1)?)?;
            total.checked_add(idf.checked_mul(frequency.min(8))?)
        }) else {
            continue;
        };
        if score > 0 {
            scored.push((score, entry.id.clone()));
        }
    }
    scored.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
    let Some((best, _)) = scored.first() else {
        return Ok(RankedSelection::Lexical(Vec::new()));
    };
    let confident = scored
        .get(1)
        .is_none_or(|(second, _)| best.saturating_sub(*second) >= 500);
    let take = if confident { 1 } else { max_expansion };
    Ok(RankedSelection::Lexical(
        scored.into_iter().take(take).map(|(_, id)| id).collect(),
    ))
}

fn indexed_terms(entry: &CatalogEntry) -> BTreeMap<String, u32> {
    let mut terms = BTreeMap::new();
    for value in std::iter::once(entry.id.as_str())
        .chain(std::iter::once(entry.name.as_str()))
        .chain(entry.metadata.namespace.iter().map(String::as_str))
        .chain(entry.metadata.aliases.iter().map(String::as_str))
    {
        for term in tokenize(value) {
            *terms.entry(term).or_default() += 1;
        }
    }
    terms
}

fn tokenize(value: &str) -> Vec<String> {
    value
        .char_indices()
        .take_while(|(index, _)| *index < MAX_DISCOVERY_QUERY_BYTES)
        .map(|(_, character)| character)
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .take(MAX_QUERY_TERMS)
        .map(str::to_owned)
        .collect()
}

fn normalize_phrase(value: &str) -> String {
    value
        .trim()
        .char_indices()
        .take_while(|(index, _)| *index < MAX_DISCOVERY_QUERY_BYTES)
        .map(|(_, character)| character)
        .map(|character| character.to_ascii_lowercase())
        .collect()
}

fn validate_stable_identifier(label: &str, value: &str, tool_id: &str) -> Result<(), HarnessError> {
    let valid = !value.is_empty()
        && value.len() <= MAX_DISCOVERY_IDENTIFIER_BYTES
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || matches!(byte, b'.' | b'_' | b'-' | b'/')
        })
        && value.as_bytes()[0].is_ascii_alphanumeric();
    if !valid {
        return Err(HarnessError::InvalidTool(format!(
            "tool {tool_id} discovery {label} must be a stable lowercase identifier"
        )));
    }
    Ok(())
}

fn fingerprint_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Tool, ToolResult};
    use async_trait::async_trait;
    use serde_json::{json, Value};
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;

    struct TestTool(ToolDefinition);

    #[async_trait]
    impl Tool for TestTool {
        fn definition(&self) -> &ToolDefinition {
            &self.0
        }

        async fn execute(
            &self,
            _: Value,
            _: CancellationToken,
        ) -> Result<ToolResult, HarnessError> {
            Ok(ToolResult::success(json!({"ok": true})))
        }
    }

    fn definition(id: &str, description: &str) -> ToolDefinition {
        ToolDefinition::new(
            id,
            id.replace('.', " "),
            description,
            json!({"type": "object"}),
        )
        .with_allowed_callers([ToolCaller::Direct, ToolCaller::DeclarativePlan])
    }

    fn register(
        registry: &mut ToolRegistry,
        id: &str,
        description: &str,
        metadata: ToolDiscoveryMetadata,
    ) {
        registry
            .register_with_discovery(Arc::new(TestTool(definition(id, description))), metadata)
            .unwrap();
    }

    fn allowlist(count: usize) -> Vec<String> {
        (0..count).map(|index| format!("tool.{index:04}")).collect()
    }

    fn scope(
        registry: &ToolRegistry,
        query: &str,
        allowlist: &[String],
        limits: ToolDiscoveryLimits,
        provider: ProviderCapabilityLimits,
    ) -> Result<(ToolScope, ToolDiscoveryStats), HarnessError> {
        registry.select_scope(query, allowlist, ToolCaller::Direct, limits, &provider)
    }

    #[test]
    fn exact_selection_is_stable_for_large_catalogs() {
        for count in [30, 100, 1_000] {
            let mut registry = ToolRegistry::default();
            for id in allowlist(count) {
                register(
                    &mut registry,
                    &id,
                    "unindexed description",
                    ToolDiscoveryMetadata::deferred(),
                );
            }
            let ids = allowlist(count);
            let target = ids[count - 1].clone();
            let (selected, stats) = scope(
                &registry,
                &target,
                &ids,
                ToolDiscoveryLimits::new().with_max_tools(4),
                ProviderCapabilityLimits::new(),
            )
            .unwrap();
            assert_eq!(selected.definitions().len(), 1);
            assert!(selected.contains(&target));
            assert_eq!(stats.candidate_count, count as u32);
            assert!(stats.catalog_exceeded_budget);
        }
    }

    #[test]
    fn guarded_large_catalog_traversal_cancels_without_warming_the_index() {
        let count = 1_000;
        let mut registry = ToolRegistry::default();
        for id in allowlist(count) {
            register(
                &mut registry,
                &id,
                "description",
                ToolDiscoveryMetadata::deferred(),
            );
        }
        let mut checkpoints = 0;
        let result = registry.select_scope_guarded(
            "tool 0999",
            &allowlist(count),
            ToolCaller::Direct,
            ToolDiscoveryLimits::new().with_max_tools(1),
            &ProviderCapabilityLimits::new(),
            &mut || {
                checkpoints += 1;
                if checkpoints == 20 {
                    Err(HarnessError::Cancelled)
                } else {
                    Ok(())
                }
            },
        );
        assert!(matches!(result, Err(HarnessError::Cancelled)));
        assert_eq!(checkpoints, 20);
        assert!(!registry.catalog_index().1);
    }

    #[test]
    fn namespace_exact_match_and_ambiguous_aliases_fail_closed() {
        let mut registry = ToolRegistry::default();
        register(
            &mut registry,
            "weather.current",
            "secret-description",
            ToolDiscoveryMetadata::deferred()
                .with_namespace("weather")
                .with_aliases(["forecast"]),
        );
        register(
            &mut registry,
            "weather.future",
            "secret-description",
            ToolDiscoveryMetadata::deferred()
                .with_namespace("weather")
                .with_aliases(["forecast"]),
        );
        let ids = vec!["weather.current".into(), "weather.future".into()];
        let tight = ToolDiscoveryLimits::new().with_max_tools(1);
        assert!(matches!(
            scope(
                &registry,
                "weather",
                &ids,
                tight,
                ProviderCapabilityLimits::new()
            ),
            Err(HarnessError::ResourceLimit(_))
        ));
        let (ambiguous, _) = scope(
            &registry,
            "forecast",
            &ids,
            tight,
            ProviderCapabilityLimits::new(),
        )
        .unwrap();
        assert!(ambiguous.is_empty());
    }

    #[test]
    fn exact_name_precedes_lexical_ranking_and_low_margin_expansion_is_bounded() {
        let mut registry = ToolRegistry::default();
        for index in 0..10 {
            let id = format!("weather.tool.{index:02}");
            register(
                &mut registry,
                &id,
                "description",
                ToolDiscoveryMetadata::deferred(),
            );
        }
        let ids = (0..10)
            .rev()
            .map(|index| format!("weather.tool.{index:02}"))
            .collect::<Vec<_>>();
        let limits = ToolDiscoveryLimits::new()
            .with_max_tools(4)
            .with_max_expansion_tools(3);
        let (exact_name, _) = scope(
            &registry,
            "weather tool 07",
            &ids,
            limits,
            ProviderCapabilityLimits::new(),
        )
        .unwrap();
        assert_eq!(exact_name.definitions().len(), 1);
        assert!(exact_name.contains("weather.tool.07"));

        let (expanded, _) = scope(
            &registry,
            "weather",
            &ids,
            limits,
            ProviderCapabilityLimits::new(),
        )
        .unwrap();
        assert_eq!(expanded.definitions().len(), 3);
        assert!(expanded.contains("weather.tool.00"));
        assert!(expanded.contains("weather.tool.01"));
        assert!(expanded.contains("weather.tool.02"));
    }

    #[test]
    fn hidden_and_caller_incompatible_tools_cannot_change_scope_local_ranking() {
        fn permitted_registry() -> ToolRegistry {
            let mut registry = ToolRegistry::default();
            for (id, alias) in [
                ("permitted.alpha", "rare"),
                ("permitted.beta", "common"),
                ("permitted.neutral", "neutral"),
            ] {
                register(
                    &mut registry,
                    id,
                    "description",
                    ToolDiscoveryMetadata::deferred().with_aliases([alias]),
                );
            }
            registry
        }

        let baseline = permitted_registry();
        let mut polluted = permitted_registry();
        let mut polluted_allowlist = vec![
            "permitted.alpha".into(),
            "permitted.beta".into(),
            "permitted.neutral".into(),
        ];
        for index in 0..100 {
            let hidden_id = format!("hidden.tool.{index:03}");
            let hidden = ToolDefinition::new(
                &hidden_id,
                "rare rare rare rare rare rare rare rare",
                "description",
                json!({"type": "object"}),
            )
            .with_allowed_callers([ToolCaller::Direct, ToolCaller::DeclarativePlan]);
            polluted
                .register_with_discovery(
                    Arc::new(TestTool(hidden)),
                    ToolDiscoveryMetadata::deferred()
                        .with_aliases([format!("rare-hidden-{index:03}")]),
                )
                .unwrap();

            let incompatible_id = format!("incompatible.tool.{index:03}");
            let incompatible = ToolDefinition::new(
                &incompatible_id,
                "rare rare rare rare rare rare rare rare",
                "description",
                json!({"type": "object"}),
            )
            .with_allowed_callers([ToolCaller::Programmatic]);
            polluted
                .register_with_discovery(
                    Arc::new(TestTool(incompatible)),
                    ToolDiscoveryMetadata::deferred()
                        .with_aliases([format!("rare-incompatible-{index:03}")]),
                )
                .unwrap();
            polluted_allowlist.push(incompatible_id);
        }

        let baseline_allowlist = vec![
            "permitted.alpha".into(),
            "permitted.beta".into(),
            "permitted.neutral".into(),
        ];
        let limits = ToolDiscoveryLimits::new()
            .with_max_tools(2)
            .with_max_expansion_tools(2);
        for caller in [ToolCaller::Direct, ToolCaller::DeclarativePlan] {
            let (baseline_scope, baseline_stats) = baseline
                .select_scope(
                    "rare common",
                    &baseline_allowlist,
                    caller,
                    limits,
                    &ProviderCapabilityLimits::new(),
                )
                .unwrap();
            let (polluted_scope, polluted_stats) = polluted
                .select_scope(
                    "rare common",
                    &polluted_allowlist,
                    caller,
                    limits,
                    &ProviderCapabilityLimits::new(),
                )
                .unwrap();
            let selected_ids = |scope: &ToolScope| {
                scope
                    .definitions()
                    .iter()
                    .map(|tool| tool.id.clone())
                    .collect::<Vec<_>>()
            };
            assert_eq!(selected_ids(&baseline_scope), selected_ids(&polluted_scope));
            assert_eq!(baseline_stats.candidate_count, 3);
            assert_eq!(polluted_stats.candidate_count, 3);
            assert_eq!(baseline_stats.selected_count, 2);
            assert_eq!(polluted_stats.selected_count, 2);
            assert_eq!(baseline_stats.deferred_candidate_count, 3);
            assert_eq!(polluted_stats.deferred_candidate_count, 3);
            assert_eq!(
                baseline_stats.catalog_exceeded_budget,
                polluted_stats.catalog_exceeded_budget
            );
            assert_eq!(baseline_stats.cache_hit, polluted_stats.cache_hit);
        }
    }

    #[test]
    fn budgets_are_exact_and_zero_provider_capacity_is_a_no_tool_scope() {
        let mut registry = ToolRegistry::default();
        register(
            &mut registry,
            "weather.current",
            "weather",
            ToolDiscoveryMetadata::hot(),
        );
        let ids = vec!["weather.current".into()];
        let definitions = registry.allowed_catalog(&ids, ToolCaller::Direct);
        let bytes = crate::limits::serialized_len(
            &definitions
                .iter()
                .map(|entry| entry.definition.clone())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let exact = ToolDiscoveryLimits::new()
            .with_max_tools(1)
            .with_max_tool_schema_bytes(bytes);
        assert_eq!(
            scope(
                &registry,
                "weather.current",
                &ids,
                exact,
                ProviderCapabilityLimits::new()
            )
            .unwrap()
            .0
            .definitions()
            .len(),
            1
        );
        assert!(scope(
            &registry,
            "weather.current",
            &ids,
            exact.with_max_tool_schema_bytes(bytes - 1),
            ProviderCapabilityLimits::new()
        )
        .is_err());
        let (empty, _) = scope(
            &registry,
            "weather.current",
            &ids,
            exact,
            ProviderCapabilityLimits::new().with_max_tools(0),
        )
        .unwrap();
        assert!(empty.is_empty());

        for provider_bytes in [0, 1] {
            let (empty, stats) = scope(
                &registry,
                "weather.current",
                &ids,
                exact,
                ProviderCapabilityLimits::new()
                    .with_max_tools(1)
                    .with_max_tool_schema_bytes(provider_bytes),
            )
            .unwrap();
            assert!(empty.is_empty());
            assert!(!stats.cache_hit);
        }

        let empty_registry = ToolRegistry::default();
        let (empty, stats) = scope(
            &empty_registry,
            "anything",
            &[],
            ToolDiscoveryLimits::new().with_max_tool_schema_bytes(1),
            ProviderCapabilityLimits::new(),
        )
        .unwrap();
        assert!(empty.is_empty());
        assert!(!stats.cache_hit);
        assert!(!empty_registry.catalog_index().1);

        let mut full_fit_registry = ToolRegistry::default();
        register(
            &mut full_fit_registry,
            "weather.current",
            "weather",
            ToolDiscoveryMetadata::hot(),
        );
        let (full, stats) = scope(
            &full_fit_registry,
            "weather.current",
            &["weather.current".into()],
            ToolDiscoveryLimits::new(),
            ProviderCapabilityLimits::new(),
        )
        .unwrap();
        assert_eq!(full.definitions().len(), 1);
        assert!(!stats.cache_hit);
        assert!(!full_fit_registry.catalog_index().1);
    }

    #[test]
    fn fingerprint_and_cache_use_only_canonical_safe_metadata() {
        let metadata = ToolDiscoveryMetadata::deferred()
            .with_namespace("weather")
            .with_aliases(["forecast", "temperature"]);
        let mut first = ToolRegistry::default();
        register(&mut first, "weather.current", "SECRET-A", metadata.clone());
        register(
            &mut first,
            "weather.future",
            "SECRET-C",
            ToolDiscoveryMetadata::deferred(),
        );
        let ids = vec!["weather.current".into(), "weather.future".into()];
        let (_, cold_stats) = scope(
            &first,
            "SECRET-A",
            &ids,
            ToolDiscoveryLimits::new().with_max_tools(1),
            ProviderCapabilityLimits::new(),
        )
        .unwrap();
        assert!(!cold_stats.cache_hit);
        let (_, warm_stats) = scope(
            &first,
            "temperature",
            &ids,
            ToolDiscoveryLimits::new().with_max_tools(1),
            ProviderCapabilityLimits::new(),
        )
        .unwrap();
        assert!(warm_stats.cache_hit);
        let first_fingerprint = first.catalog_fingerprint();

        let mut second = ToolRegistry::default();
        register(&mut second, "weather.current", "SECRET-B", metadata);
        register(
            &mut second,
            "weather.future",
            "SECRET-D",
            ToolDiscoveryMetadata::deferred(),
        );
        assert_eq!(first_fingerprint, second.catalog_fingerprint());

        let duplicate = first.register_with_discovery(
            Arc::new(TestTool(definition("weather.current", "duplicate"))),
            ToolDiscoveryMetadata::deferred(),
        );
        assert!(duplicate.is_err());
        let invalid_metadata = first.register_with_discovery(
            Arc::new(TestTool(definition("weather.invalid", "invalid"))),
            ToolDiscoveryMetadata::deferred().with_namespace("not stable"),
        );
        assert!(invalid_metadata.is_err());
        assert_eq!(first_fingerprint, first.catalog_fingerprint());
        let (_, preserved_stats) = scope(
            &first,
            "temperature",
            &ids,
            ToolDiscoveryLimits::new().with_max_tools(1),
            ProviderCapabilityLimits::new(),
        )
        .unwrap();
        assert!(preserved_stats.cache_hit);

        register(
            &mut first,
            "weather.archive",
            "SECRET-E",
            ToolDiscoveryMetadata::deferred(),
        );
        assert_ne!(first_fingerprint, first.catalog_fingerprint());

        let mut permuted = ToolRegistry::default();
        register(
            &mut permuted,
            "weather.archive",
            "SECRET-W",
            ToolDiscoveryMetadata::deferred(),
        );
        register(
            &mut permuted,
            "weather.future",
            "SECRET-X",
            ToolDiscoveryMetadata::deferred(),
        );
        register(
            &mut permuted,
            "weather.current",
            "SECRET-Y",
            ToolDiscoveryMetadata::deferred()
                .with_namespace("weather")
                .with_aliases(["temperature", "forecast"]),
        );
        assert_eq!(first.catalog_fingerprint(), permuted.catalog_fingerprint());
    }

    #[test]
    fn discovery_metadata_validation_rejects_unstable_or_duplicate_values() {
        let mut registry = ToolRegistry::default();
        assert!(registry
            .register_with_discovery(
                Arc::new(TestTool(definition("tool", "description"))),
                ToolDiscoveryMetadata::deferred().with_namespace("Not Stable"),
            )
            .is_err());
        assert!(registry
            .register_with_discovery(
                Arc::new(TestTool(definition("tool-2", "description"))),
                ToolDiscoveryMetadata::deferred().with_aliases(["same", "same"]),
            )
            .is_err());
        assert!(registry
            .register_with_discovery(
                Arc::new(TestTool(definition("tool-3", "description"))),
                ToolDiscoveryMetadata::deferred().with_aliases(["same-token", "same_token"]),
            )
            .is_err());
    }

    #[test]
    fn registration_bounds_are_canonical_and_failures_preserve_warm_catalog() {
        use crate::tool::{MAX_TOOL_ID_BYTES, MAX_TOOL_NAME_BYTES, MAX_TOOL_SAFE_METADATA_BYTES};

        let mut registry = ToolRegistry::default();
        register(
            &mut registry,
            "weather.current",
            "description",
            ToolDiscoveryMetadata::deferred(),
        );
        let allowlist = vec!["weather.current".into()];
        let limits = ToolDiscoveryLimits::new().with_max_tools(0);
        let (_, cold) = scope(
            &registry,
            "weather",
            &allowlist,
            limits,
            ProviderCapabilityLimits::new(),
        )
        .unwrap();
        assert!(!cold.cache_hit);
        let fingerprint = registry.catalog_fingerprint();
        let (_, warm) = scope(
            &registry,
            "weather",
            &allowlist,
            ToolDiscoveryLimits::new().with_max_tools(1),
            ProviderCapabilityLimits::new().with_max_tools(0),
        )
        .unwrap();
        assert!(!warm.cache_hit);

        let exact_id = format!("a{}", "x".repeat(MAX_TOOL_ID_BYTES - 1));
        let exact_name = "N".repeat(MAX_TOOL_NAME_BYTES);
        registry
            .register_with_discovery(
                Arc::new(TestTool(ToolDefinition::new(
                    &exact_id,
                    &exact_name,
                    "description",
                    json!({"type": "object"}),
                ))),
                ToolDiscoveryMetadata::deferred()
                    .with_namespace("n".repeat(MAX_DISCOVERY_IDENTIFIER_BYTES))
                    .with_aliases(["a".repeat(MAX_DISCOVERY_IDENTIFIER_BYTES)]),
            )
            .unwrap();
        let fingerprint_after_valid = registry.catalog_fingerprint();
        assert_ne!(fingerprint, fingerprint_after_valid);
        let all_ids = vec!["weather.current".into(), exact_id];
        let (_, warmed) = scope(
            &registry,
            "no match",
            &all_ids,
            ToolDiscoveryLimits::new().with_max_tools(1),
            ProviderCapabilityLimits::new(),
        )
        .unwrap();
        assert!(warmed.cache_hit);

        let invalid_definitions = [
            ToolDefinition::new(
                format!("a{}", "x".repeat(MAX_TOOL_ID_BYTES)),
                "Overlong id",
                "description",
                json!({"type": "object"}),
            ),
            ToolDefinition::new(
                "invalid.name.length",
                "N".repeat(MAX_TOOL_NAME_BYTES + 1),
                "description",
                json!({"type": "object"}),
            ),
            ToolDefinition::new(
                " leading",
                "Leading id whitespace",
                "description",
                json!({"type": "object"}),
            ),
            ToolDefinition::new(
                "unicode.\u{e9}",
                "Unicode id",
                "description",
                json!({"type": "object"}),
            ),
            ToolDefinition::new(
                "unicode.decomposed",
                "Cafe\u{301}",
                "description",
                json!({"type": "object"}),
            ),
            ToolDefinition::new(
                "unicode.composed",
                "Caf\u{e9}",
                "description",
                json!({"type": "object"}),
            ),
            ToolDefinition::new(
                "control.name",
                "Control\nName",
                "description",
                json!({"type": "object"}),
            ),
            ToolDefinition::new(
                "bidi.name",
                "Bidi \u{202e}Name",
                "description",
                json!({"type": "object"}),
            ),
        ];
        for definition in invalid_definitions {
            assert!(registry
                .register_with_discovery(
                    Arc::new(TestTool(definition)),
                    ToolDiscoveryMetadata::deferred(),
                )
                .is_err());
        }

        for metadata in [
            ToolDiscoveryMetadata::deferred()
                .with_namespace(format!("n{}", "x".repeat(MAX_DISCOVERY_IDENTIFIER_BYTES))),
            ToolDiscoveryMetadata::deferred()
                .with_aliases([format!("a{}", "x".repeat(MAX_DISCOVERY_IDENTIFIER_BYTES))]),
            ToolDiscoveryMetadata::deferred().with_namespace("unicode.\u{e9}"),
            ToolDiscoveryMetadata::deferred().with_aliases(["control\u{0000}"]),
        ] {
            assert!(registry
                .register_with_discovery(
                    Arc::new(TestTool(definition("invalid.metadata", "description"))),
                    metadata,
                )
                .is_err());
        }

        let stuffed_aliases = (0..32)
            .map(|index| {
                format!(
                    "a{index:02}{}",
                    "x".repeat(MAX_DISCOVERY_IDENTIFIER_BYTES - 3)
                )
            })
            .collect::<Vec<_>>();
        let stuffed = registry
            .register_with_discovery(
                Arc::new(TestTool(definition("stuffed.metadata", "description"))),
                ToolDiscoveryMetadata::deferred().with_aliases(stuffed_aliases),
            )
            .unwrap_err();
        assert!(matches!(
            stuffed,
            HarnessError::InvalidTool(message)
                if message.contains("safe discovery metadata")
                    && message.contains(&MAX_TOOL_SAFE_METADATA_BYTES.to_string())
        ));

        assert_eq!(fingerprint_after_valid, registry.catalog_fingerprint());
        let (_, preserved) = scope(
            &registry,
            "no match",
            &all_ids,
            ToolDiscoveryLimits::new().with_max_tools(1),
            ProviderCapabilityLimits::new(),
        )
        .unwrap();
        assert!(preserved.cache_hit);
    }

    #[test]
    fn bounded_malicious_unicode_query_is_safe_and_does_not_search_payload_metadata() {
        let mut registry = ToolRegistry::default();
        register(
            &mut registry,
            "weather.current",
            "payload-secret schema-secret provider-secret",
            ToolDiscoveryMetadata::deferred().with_aliases(["temperature"]),
        );
        let ids = vec!["weather.current".into()];
        let query = format!("{}payload-secret", "🦙".repeat(8_000));
        let (selected, _) = scope(
            &registry,
            &query,
            &ids,
            ToolDiscoveryLimits::new().with_max_tools(0),
            ProviderCapabilityLimits::new(),
        )
        .unwrap();
        assert!(selected.is_empty());
    }

    #[test]
    fn exact_matches_are_not_disabled_by_lexical_expansion_limit() {
        let mut registry = ToolRegistry::default();
        for id in ["weather.current", "weather.future"] {
            register(
                &mut registry,
                id,
                "description",
                ToolDiscoveryMetadata::deferred(),
            );
        }
        let ids = vec!["weather.current".into(), "weather.future".into()];
        let (selected, _) = scope(
            &registry,
            "weather.current",
            &ids,
            ToolDiscoveryLimits::new()
                .with_max_tools(1)
                .with_max_expansion_tools(0),
            ProviderCapabilityLimits::new(),
        )
        .unwrap();
        assert!(selected.contains("weather.current"));
    }
}
