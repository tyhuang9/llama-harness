use crate::{HarnessError, ProviderCapabilityLimits, ToolCaller, ToolDefinition, ToolRegistry};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::Arc,
};

/// Version of the canonical safe-metadata catalog fingerprint format.
pub const CATALOG_FINGERPRINT_VERSION: u32 = 1;
const MAX_DISCOVERY_QUERY_BYTES: usize = 4096;
const MAX_QUERY_TERMS: usize = 64;

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
            if !unique.insert(alias) {
                return Err(HarnessError::InvalidTool(format!(
                    "tool {tool_id} has duplicate discovery aliases"
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
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct CatalogFingerprint {
    /// Canonical fingerprint format version.
    pub version: u32,
    /// Lowercase hexadecimal BLAKE3 digest.
    pub digest: String,
}

/// Immutable set of tools selected for one caller and one run.
#[derive(Clone, Debug)]
pub struct ToolScope {
    caller: ToolCaller,
    tool_ids: BTreeSet<String>,
    definitions: Arc<[ToolDefinition]>,
}

impl ToolScope {
    /// Returns the execution caller this scope was selected for.
    pub fn caller(&self) -> ToolCaller {
        self.caller
    }

    /// Returns whether the scope admits the exact registered tool ID.
    pub fn contains(&self, tool_id: &str) -> bool {
        self.tool_ids.contains(tool_id)
    }

    /// Returns selected definitions in deterministic allowlist order.
    pub fn definitions(&self) -> &[ToolDefinition] {
        &self.definitions
    }

    /// Returns the number of selected definitions.
    pub fn len(&self) -> usize {
        self.definitions.len()
    }

    /// Returns whether no tools were selected.
    pub fn is_empty(&self) -> bool {
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
    pub(crate) serialized_bytes: u64,
}

pub(crate) struct CatalogIndex {
    entries: Vec<CatalogEntry>,
    document_frequency: BTreeMap<String, u32>,
    fingerprint: CatalogFingerprint,
}

impl CatalogIndex {
    pub(crate) fn build(mut entries: Vec<CatalogEntry>) -> Self {
        entries.sort_by(|left, right| left.id.cmp(&right.id));
        let mut document_frequency = BTreeMap::<String, u32>::new();
        for entry in &mut entries {
            entry.terms = indexed_terms(entry);
            for term in entry.terms.keys() {
                *document_frequency.entry(term.clone()).or_default() += 1;
            }
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"llama-harness-tool-catalog\0v1\0");
        for entry in &entries {
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
        Self {
            entries,
            document_frequency,
            fingerprint: CatalogFingerprint {
                version: CATALOG_FINGERPRINT_VERSION,
                digest: hasher.finalize().to_hex().to_string(),
            },
        }
    }

    pub(crate) fn fingerprint(&self) -> CatalogFingerprint {
        self.fingerprint.clone()
    }
}

impl ToolRegistry {
    /// Returns the versioned fingerprint for the cached safe-metadata catalog.
    pub fn catalog_fingerprint(&self) -> CatalogFingerprint {
        self.catalog_index().0.fingerprint()
    }

    pub(crate) fn select_scope(
        &self,
        query: &str,
        allowlist: &[String],
        caller: ToolCaller,
        host_limits: ToolDiscoveryLimits,
        provider_limits: &ProviderCapabilityLimits,
    ) -> Result<(ToolScope, ToolDiscoveryStats), HarnessError> {
        let limits = host_limits.effective(provider_limits);
        let allowed = self.allowed_catalog(allowlist, caller);
        let deferred_count = allowed
            .iter()
            .filter(|entry| entry.metadata.exposure == ToolExposure::Deferred)
            .count();
        let (index, cache_hit) = self.catalog_index();
        let base_stats = ToolDiscoveryStats {
            candidate_count: allowed.len() as u32,
            selected_count: 0,
            deferred_candidate_count: deferred_count as u32,
            catalog_exceeded_budget: false,
            cache_hit,
        };

        if limits.max_tools == 0 || limits.max_bytes == 0 {
            return Ok((ToolScope::empty(caller), base_stats));
        }
        let all_definitions = allowed
            .iter()
            .map(|entry| entry.definition.clone())
            .collect::<Vec<_>>();
        if fits(&allowed, limits)? {
            return Ok(scope_with_stats(caller, all_definitions, base_stats, false));
        }

        let allowed_by_id = allowed
            .iter()
            .map(|entry| {
                (
                    entry.definition.id.as_str(),
                    (entry.definition, entry.metadata),
                )
            })
            .collect::<HashMap<_, _>>();
        let mut required = allowed
            .iter()
            .filter(|entry| entry.metadata.exposure == ToolExposure::Hot)
            .map(|entry| entry.definition.id.clone())
            .collect::<BTreeSet<_>>();
        let ranked = rank(&index, query, &allowed_by_id, limits.max_expansion as usize);
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
        entries.iter().map(|entry| entry.serialized_bytes),
        limits,
    )
}

fn fits_refs(
    entries: &[&AllowedCatalogEntry<'_>],
    limits: EffectiveLimits,
) -> Result<bool, HarnessError> {
    fits_lengths(
        entries.len(),
        entries.iter().map(|entry| entry.serialized_bytes),
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
    index: &CatalogIndex,
    query: &str,
    allowed: &HashMap<&str, (&ToolDefinition, &ToolDiscoveryMetadata)>,
    max_expansion: usize,
) -> RankedSelection {
    let normalized = normalize_phrase(query);
    let query_terms = tokenize(query);
    let exact_id = index
        .entries
        .iter()
        .filter(|entry| allowed.contains_key(entry.id.as_str()))
        .filter(|entry| entry.id == normalized)
        .map(|entry| entry.id.clone())
        .collect::<Vec<_>>();
    if exact_id.len() == 1 {
        return RankedSelection::Exact(exact_id);
    }
    let exact_name = index
        .entries
        .iter()
        .filter(|entry| allowed.contains_key(entry.id.as_str()))
        .filter(|entry| normalize_phrase(&entry.name) == normalized)
        .map(|entry| entry.id.clone())
        .collect::<Vec<_>>();
    if exact_name.len() == 1 {
        return RankedSelection::Exact(exact_name);
    }
    if exact_name.len() > 1 {
        return RankedSelection::Exact(Vec::new());
    }
    let exact_namespace = index
        .entries
        .iter()
        .filter(|entry| allowed.contains_key(entry.id.as_str()))
        .filter(|entry| entry.metadata.namespace.as_deref() == Some(normalized.as_str()))
        .map(|entry| entry.id.clone())
        .collect::<Vec<_>>();
    if !exact_namespace.is_empty() {
        return RankedSelection::Exact(exact_namespace);
    }
    let exact_alias = index
        .entries
        .iter()
        .filter(|entry| allowed.contains_key(entry.id.as_str()))
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
        return RankedSelection::Exact(exact_alias);
    }
    if exact_alias.len() > 1 {
        return RankedSelection::Exact(Vec::new());
    }
    if max_expansion == 0 {
        return RankedSelection::Lexical(Vec::new());
    }
    if query_terms.is_empty() {
        return RankedSelection::Lexical(Vec::new());
    }
    let documents = index.entries.len() as u64;
    let mut scored = index
        .entries
        .iter()
        .filter(|entry| allowed.contains_key(entry.id.as_str()))
        .filter_map(|entry| {
            let score = query_terms.iter().try_fold(0u64, |total, term| {
                let Some(frequency) = entry.terms.get(term).copied().map(u64::from) else {
                    return Some(total);
                };
                let document_frequency = u64::from(*index.document_frequency.get(term)?);
                let idf = documents
                    .checked_add(1)?
                    .checked_mul(1_000)?
                    .checked_div(document_frequency.checked_add(1)?)?;
                total.checked_add(idf.checked_mul(frequency.min(8))?)
            })?;
            (score > 0).then(|| (score, entry.id.clone()))
        })
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
    let Some((best, _)) = scored.first() else {
        return RankedSelection::Lexical(Vec::new());
    };
    let confident = scored
        .get(1)
        .is_none_or(|(second, _)| best.saturating_sub(*second) >= 500);
    let take = if confident { 1 } else { max_expansion };
    RankedSelection::Lexical(scored.into_iter().take(take).map(|(_, id)| id).collect())
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
        && value.len() <= 128
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
            assert_eq!(selected.len(), 1);
            assert!(selected.contains(&target));
            assert_eq!(stats.candidate_count, count as u32);
            assert!(stats.catalog_exceeded_budget);
        }
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
        assert_eq!(exact_name.len(), 1);
        assert!(exact_name.contains("weather.tool.07"));

        let (expanded, _) = scope(
            &registry,
            "weather",
            &ids,
            limits,
            ProviderCapabilityLimits::new(),
        )
        .unwrap();
        assert_eq!(expanded.len(), 3);
        assert!(expanded.contains("weather.tool.00"));
        assert!(expanded.contains("weather.tool.01"));
        assert!(expanded.contains("weather.tool.02"));
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
    }

    #[test]
    fn fingerprint_and_cache_use_only_canonical_safe_metadata() {
        let metadata = ToolDiscoveryMetadata::deferred()
            .with_namespace("weather")
            .with_aliases(["forecast", "temperature"]);
        let mut first = ToolRegistry::default();
        register(&mut first, "weather.current", "SECRET-A", metadata.clone());
        let first_fingerprint = first.catalog_fingerprint();
        let (_, first_stats) = scope(
            &first,
            "SECRET-A",
            &["weather.current".into()],
            ToolDiscoveryLimits::new().with_max_tools(0),
            ProviderCapabilityLimits::new(),
        )
        .unwrap();
        assert!(first_stats.cache_hit);

        let mut second = ToolRegistry::default();
        register(&mut second, "weather.current", "SECRET-B", metadata);
        assert_eq!(first_fingerprint, second.catalog_fingerprint());

        register(
            &mut first,
            "weather.future",
            "SECRET-C",
            ToolDiscoveryMetadata::deferred(),
        );
        assert_ne!(first_fingerprint, first.catalog_fingerprint());

        let mut permuted = ToolRegistry::default();
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
