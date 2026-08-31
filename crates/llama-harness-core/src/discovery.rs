use crate::{
    runner::check_stopped, HarnessError, PreparedToolCatalog, ProviderCapabilityLimits, ToolCaller,
    ToolDefinition, ToolRegistry,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

/// Version of the canonical safe-metadata catalog fingerprint format.
pub const CATALOG_FINGERPRINT_VERSION: u32 = 1;
const MAX_DISCOVERY_QUERY_BYTES: usize = 4096;
const MAX_QUERY_TERMS: usize = 64;
// BM25 scores use nine decimal fixed-point digits. IDF is the standard
// ln(1 + (N - df + 0.5) / (df + 0.5)); its rational form is reduced to [1, 2)
// and evaluated with 20 atanh-series terms. On that interval the omitted
// series tail is below 2e-19 before fixed-point rounding. Every division rounds
// half up; saturation is reserved for unreachable integer-capacity limits.
const BM25_SCALE: u128 = 1_000_000_000;
const BM25_K1: u128 = 1_200_000_000;
const BM25_B: u128 = 750_000_000;
const LN_2_SCALED: u128 = 693_147_181;
const LN_SERIES_TERMS: u32 = 20;
const MIN_CONFIDENCE_MARGIN: u64 = 500_000_000;
pub(crate) const MAX_DISCOVERY_IDENTIFIER_BYTES: usize = 128;
pub(crate) const DISCOVERY_GUARD_INTERVAL: usize = 64;
pub(crate) const MAX_SCOPE_CATALOG_CACHE_ENTRIES: usize = 16;
pub(crate) const MAX_PREPARED_CATALOG_CACHE_ENTRIES: usize = 32;
pub(crate) const MAX_SCOPE_CATALOG_CACHE_BYTES: usize = 16 * 1024 * 1024;
pub(crate) const MAX_PREPARED_CATALOG_CACHE_BYTES: usize = 8 * 1024 * 1024;
pub(crate) const MAX_DEFERRED_LEXICAL_TERMS: usize = 256;
pub(crate) const MAX_DEFERRED_LEXICAL_BYTES: usize = 1024;
const TREE_NODE_OVERHEAD: usize = 4 * std::mem::size_of::<usize>();
const ARC_ALLOCATION_OVERHEAD: usize = 2 * std::mem::size_of::<usize>();

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

    pub(crate) fn validate_deferred_lexical_shape(
        &self,
        tool_id: &str,
        tool_name: &str,
    ) -> Result<(), HarnessError> {
        if self.exposure != ToolExposure::Deferred {
            return Ok(());
        }
        let mut term_count = 0usize;
        let mut lexical_bytes = 0usize;
        for value in std::iter::once(tool_id)
            .chain(std::iter::once(tool_name))
            .chain(self.namespace.iter().map(String::as_str))
            .chain(self.aliases.iter().map(String::as_str))
        {
            for term in tokenize(value) {
                term_count = term_count.saturating_add(1);
                lexical_bytes = lexical_bytes.saturating_add(term.len());
                if term_count > MAX_DEFERRED_LEXICAL_TERMS
                    || lexical_bytes > MAX_DEFERRED_LEXICAL_BYTES
                {
                    return Err(HarnessError::InvalidTool(format!(
                        "tool {tool_id} deferred discovery terms exceed the lexical index limit"
                    )));
                }
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
    prepared: Option<Arc<PreparedToolCatalog>>,
}

impl ToolScope {
    pub(crate) fn caller(&self) -> ToolCaller {
        self.caller
    }

    pub(crate) fn contains(&self, tool_id: &str) -> bool {
        self.tool_ids.contains(tool_id)
    }

    pub(crate) fn definitions(&self) -> &[ToolDefinition] {
        self.prepared
            .as_deref()
            .map_or(&[], PreparedToolCatalog::definitions)
    }

    pub(crate) fn prepared(&self) -> Option<Arc<PreparedToolCatalog>> {
        self.prepared.clone()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.prepared.is_none()
    }

    #[cfg(test)]
    pub(crate) fn serialized_definitions(&self) -> &[u8] {
        self.prepared
            .as_deref()
            .map_or(&b"[]"[..], PreparedToolCatalog::serialized_definitions)
    }

    fn empty(caller: ToolCaller) -> Self {
        Self {
            caller,
            tool_ids: BTreeSet::new(),
            prepared: None,
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
    pub(crate) document_len: u32,
}

pub(crate) struct AllowedCatalogEntry<'a> {
    pub(crate) definition: &'a ToolDefinition,
    pub(crate) metadata: &'a ToolDiscoveryMetadata,
    pub(crate) serialized_definition: &'a Arc<[u8]>,
    pub(crate) serialized_provider_tool: &'a Arc<[u8]>,
    pub(crate) version: u64,
}

pub(crate) struct CatalogIndex {
    entries: Vec<CatalogEntry>,
    postings: BTreeMap<String, Vec<(usize, u128)>>,
    total_document_len: u128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CatalogCacheKey {
    caller: ToolCaller,
    tools: Arc<[(String, u64)]>,
}

impl CatalogCacheKey {
    pub(crate) fn new(caller: ToolCaller, mut tools: Vec<(String, u64)>) -> Self {
        tools.sort();
        tools.dedup();
        Self {
            caller,
            tools: Arc::from(tools),
        }
    }

    fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(ARC_ALLOCATION_OVERHEAD)
            .saturating_add(
                self.tools
                    .len()
                    .saturating_mul(std::mem::size_of::<(String, u64)>()),
            )
            .saturating_add(
                self.tools
                    .iter()
                    .map(|(id, _)| id.capacity())
                    .fold(0usize, usize::saturating_add),
            )
    }
}

struct CatalogCacheEntry {
    key: CatalogCacheKey,
    index: Arc<CatalogIndex>,
    retained_bytes: usize,
    last_used: AtomicU64,
}

#[derive(Default)]
pub(crate) struct CatalogCache {
    entries: VecDeque<CatalogCacheEntry>,
    retained_bytes: usize,
    evictions: u64,
    clock: AtomicU64,
}

impl CatalogCache {
    pub(crate) fn get(&self, key: &CatalogCacheKey) -> Option<Arc<CatalogIndex>> {
        let entry = self.entries.iter().find(|entry| entry.key == *key)?;
        entry
            .last_used
            .fetch_max(self.next_recency(), Ordering::Relaxed);
        Some(Arc::clone(&entry.index))
    }

    pub(crate) fn insert(&mut self, key: CatalogCacheKey, index: Arc<CatalogIndex>) -> bool {
        let retained_bytes = key.retained_bytes().saturating_add(index.retained_bytes());
        if retained_bytes > MAX_SCOPE_CATALOG_CACHE_BYTES {
            return false;
        }
        while self.entries.len() >= MAX_SCOPE_CATALOG_CACHE_ENTRIES
            || self.retained_bytes.saturating_add(retained_bytes) > MAX_SCOPE_CATALOG_CACHE_BYTES
        {
            let Some(position) = self.least_recently_used_position() else {
                break;
            };
            let Some(evicted) = self.entries.remove(position) else {
                break;
            };
            self.retained_bytes = self.retained_bytes.saturating_sub(evicted.retained_bytes);
            self.evictions = self.evictions.saturating_add(1);
        }
        self.retained_bytes = self.retained_bytes.saturating_add(retained_bytes);
        self.entries.push_back(CatalogCacheEntry {
            key,
            index,
            retained_bytes,
            last_used: AtomicU64::new(self.next_recency()),
        });
        true
    }

    fn next_recency(&self) -> u64 {
        self.clock
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                Some(value.saturating_add(1))
            })
            .unwrap_or(u64::MAX)
            .saturating_add(1)
    }

    fn least_recently_used_position(&self) -> Option<usize> {
        self.entries
            .iter()
            .enumerate()
            .min_by_key(|(position, entry)| (entry.last_used.load(Ordering::Relaxed), *position))
            .map(|(position, _)| position)
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[cfg(test)]
    pub(crate) fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    #[cfg(test)]
    pub(crate) fn evictions(&self) -> u64 {
        self.evictions
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PreparedCatalogKey(Arc<[(String, u64)]>);

impl PreparedCatalogKey {
    fn new(
        entries: &[&AllowedCatalogEntry<'_>],
        guard: &mut impl FnMut() -> Result<(), HarnessError>,
    ) -> Result<Self, HarnessError> {
        let mut tools = Vec::with_capacity(entries.len());
        for (position, entry) in entries.iter().enumerate() {
            if position % DISCOVERY_GUARD_INTERVAL == 0 {
                guard()?;
            }
            tools.push((entry.definition.id.clone(), entry.version));
        }
        guard()?;
        Ok(Self(Arc::from(tools)))
    }

    fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(ARC_ALLOCATION_OVERHEAD)
            .saturating_add(
                self.0
                    .len()
                    .saturating_mul(std::mem::size_of::<(String, u64)>()),
            )
            .saturating_add(
                self.0
                    .iter()
                    .map(|(id, _)| id.capacity())
                    .fold(0usize, usize::saturating_add),
            )
    }
}

struct PreparedCatalogCacheEntry {
    key: PreparedCatalogKey,
    catalog: Arc<PreparedToolCatalog>,
    retained_bytes: usize,
    last_used: AtomicU64,
}

#[derive(Default)]
pub(crate) struct PreparedCatalogCache {
    entries: VecDeque<PreparedCatalogCacheEntry>,
    retained_bytes: usize,
    evictions: u64,
    clock: AtomicU64,
}

impl PreparedCatalogCache {
    pub(crate) fn get(&self, key: &PreparedCatalogKey) -> Option<Arc<PreparedToolCatalog>> {
        let entry = self.entries.iter().find(|entry| entry.key == *key)?;
        entry
            .last_used
            .fetch_max(self.next_recency(), Ordering::Relaxed);
        Some(Arc::clone(&entry.catalog))
    }

    pub(crate) fn insert(
        &mut self,
        key: PreparedCatalogKey,
        catalog: Arc<PreparedToolCatalog>,
    ) -> bool {
        let retained_bytes = key
            .retained_bytes()
            .saturating_add(prepared_catalog_retained_bytes(&catalog));
        if retained_bytes > MAX_PREPARED_CATALOG_CACHE_BYTES {
            return false;
        }
        while self.entries.len() >= MAX_PREPARED_CATALOG_CACHE_ENTRIES
            || self.retained_bytes.saturating_add(retained_bytes) > MAX_PREPARED_CATALOG_CACHE_BYTES
        {
            let Some(position) = self.least_recently_used_position() else {
                break;
            };
            let Some(evicted) = self.entries.remove(position) else {
                break;
            };
            self.retained_bytes = self.retained_bytes.saturating_sub(evicted.retained_bytes);
            self.evictions = self.evictions.saturating_add(1);
        }
        self.retained_bytes = self.retained_bytes.saturating_add(retained_bytes);
        self.entries.push_back(PreparedCatalogCacheEntry {
            key,
            catalog,
            retained_bytes,
            last_used: AtomicU64::new(self.next_recency()),
        });
        true
    }

    fn next_recency(&self) -> u64 {
        self.clock
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                Some(value.saturating_add(1))
            })
            .unwrap_or(u64::MAX)
            .saturating_add(1)
    }

    fn least_recently_used_position(&self) -> Option<usize> {
        self.entries
            .iter()
            .enumerate()
            .min_by_key(|(position, entry)| (entry.last_used.load(Ordering::Relaxed), *position))
            .map(|(position, _)| position)
    }

    #[cfg(test)]
    pub(crate) fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    #[cfg(test)]
    pub(crate) fn evictions(&self) -> u64 {
        self.evictions
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
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
            entry.document_len = entry
                .terms
                .values()
                .copied()
                .fold(0u32, u32::saturating_add);
        }
        let mut total_document_len = 0u128;
        for (position, entry) in entries.iter().enumerate() {
            if position % DISCOVERY_GUARD_INTERVAL == 0 {
                guard()?;
            }
            total_document_len = total_document_len.saturating_add(u128::from(entry.document_len));
        }
        let documents = entries.len() as u128;
        let mut postings = BTreeMap::<String, Vec<(usize, u128)>>::new();
        for (position, entry) in entries.iter_mut().enumerate() {
            if position % DISCOVERY_GUARD_INTERVAL == 0 {
                guard()?;
            }
            let length_normalization = bm25_length_normalization(
                documents,
                u128::from(entry.document_len),
                total_document_len,
            );
            for (term, frequency) in std::mem::take(&mut entry.terms) {
                let term_weight = bm25_term_weight(u128::from(frequency), length_normalization);
                postings
                    .entry(term)
                    .or_default()
                    .push((position, term_weight));
            }
        }
        guard()?;
        Ok(Self {
            entries,
            postings,
            total_document_len,
        })
    }

    fn retained_bytes(&self) -> usize {
        let entries = self
            .entries
            .capacity()
            .saturating_mul(std::mem::size_of::<CatalogEntry>());
        let entry_allocations = self.entries.iter().fold(0usize, |total, entry| {
            total
                .saturating_add(entry.id.capacity())
                .saturating_add(entry.name.capacity())
                .saturating_add(
                    entry
                        .metadata
                        .namespace
                        .as_ref()
                        .map_or(0, String::capacity),
                )
                .saturating_add(
                    entry
                        .metadata
                        .aliases
                        .capacity()
                        .saturating_mul(std::mem::size_of::<String>()),
                )
                .saturating_add(
                    entry
                        .metadata
                        .aliases
                        .iter()
                        .map(String::capacity)
                        .fold(0usize, usize::saturating_add),
                )
        });
        std::mem::size_of::<Self>()
            .saturating_add(ARC_ALLOCATION_OVERHEAD)
            .saturating_add(entries)
            .saturating_add(entry_allocations)
            .saturating_add(postings_retained_bytes(&self.postings))
    }
}

fn postings_retained_bytes(map: &BTreeMap<String, Vec<(usize, u128)>>) -> usize {
    map.iter().fold(0usize, |total, (term, postings)| {
        total
            .saturating_add(std::mem::size_of::<(String, Vec<(usize, u128)>)>())
            .saturating_add(TREE_NODE_OVERHEAD)
            .saturating_add(term.capacity())
            .saturating_add(
                postings
                    .capacity()
                    .saturating_mul(std::mem::size_of::<(usize, u128)>()),
            )
    })
}

fn prepared_catalog_retained_bytes(catalog: &PreparedToolCatalog) -> usize {
    let definitions = catalog
        .definitions()
        .iter()
        .fold(0usize, |total, definition| {
            total.saturating_add(tool_definition_retained_bytes(definition))
        });
    std::mem::size_of::<PreparedToolCatalog>()
        .saturating_add(ARC_ALLOCATION_OVERHEAD.saturating_mul(3))
        .saturating_add(definitions)
        .saturating_add(catalog.serialized_definitions().len())
        .saturating_add(catalog.provider_tools_json().get().len())
}

fn tool_definition_retained_bytes(definition: &ToolDefinition) -> usize {
    std::mem::size_of::<ToolDefinition>()
        .saturating_add(definition.id.capacity())
        .saturating_add(definition.name.capacity())
        .saturating_add(definition.description.capacity())
        .saturating_add(
            definition
                .concurrency_key
                .as_ref()
                .map_or(0, String::capacity),
        )
        .saturating_add(
            definition.allowed_callers.len().saturating_mul(
                std::mem::size_of::<ToolCaller>().saturating_add(TREE_NODE_OVERHEAD),
            ),
        )
        .saturating_add(json_value_retained_bytes(&definition.arguments_schema))
        .saturating_add(
            definition
                .output_schema
                .as_ref()
                .map_or(0, json_value_retained_bytes),
        )
}

fn json_value_retained_bytes(value: &serde_json::Value) -> usize {
    let base = std::mem::size_of::<serde_json::Value>();
    match value {
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => base,
        serde_json::Value::String(value) => base.saturating_add(value.capacity()),
        serde_json::Value::Array(values) => base
            .saturating_add(
                values
                    .capacity()
                    .saturating_mul(std::mem::size_of::<serde_json::Value>()),
            )
            .saturating_add(values.iter().fold(0usize, |total, value| {
                total.saturating_add(
                    json_value_retained_bytes(value)
                        .saturating_sub(std::mem::size_of::<serde_json::Value>()),
                )
            })),
        serde_json::Value::Object(values) => {
            base.saturating_add(values.iter().fold(0usize, |total, (key, value)| {
                total
                    .saturating_add(std::mem::size_of::<(String, serde_json::Value)>())
                    .saturating_add(TREE_NODE_OVERHEAD)
                    .saturating_add(key.capacity())
                    .saturating_add(
                        json_value_retained_bytes(value)
                            .saturating_sub(std::mem::size_of::<serde_json::Value>()),
                    )
            }))
        }
    }
}

pub(crate) fn fingerprint_catalog(mut entries: Vec<CatalogEntry>) -> CatalogFingerprint {
    entries.sort_by(|left, right| left.id.cmp(&right.id));
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
    CatalogFingerprint {
        version: CATALOG_FINGERPRINT_VERSION,
        digest: hasher.finalize().to_hex().to_string(),
    }
}

impl ToolRegistry {
    /// Returns the versioned fingerprint for the cached safe-metadata catalog.
    pub fn catalog_fingerprint(&self) -> CatalogFingerprint {
        self.catalog_fingerprint_cached()
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
        let mut guard = || {
            #[cfg(test)]
            if let Some(checkpoint) = &self.discovery_checkpoint {
                checkpoint(caller);
            }
            check_stopped(cancellation, deadline, "run deadline reached")
        };
        self.select_scope_guarded(
            query,
            allowlist,
            caller,
            host_limits,
            provider_limits,
            &mut guard,
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
        let mut deferred_count = 0usize;
        for (position, entry) in allowed.iter().enumerate() {
            if position % DISCOVERY_GUARD_INTERVAL == 0 {
                guard()?;
            }
            if entry.metadata.exposure == ToolExposure::Deferred {
                deferred_count = deferred_count.saturating_add(1);
            }
        }
        guard()?;
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
            guard()?;
            return Ok((ToolScope::empty(caller), base_stats));
        }
        if fits(&allowed, limits, guard)? {
            guard()?;
            let selected = collect_all(&allowed, guard)?;
            return scope_with_stats(self, caller, selected, base_stats, false, guard);
        }

        let mut required = BTreeSet::new();
        for (position, entry) in allowed.iter().enumerate() {
            if position % DISCOVERY_GUARD_INTERVAL == 0 {
                guard()?;
            }
            if entry.metadata.exposure == ToolExposure::Hot {
                required.insert(entry.definition.id.clone());
            }
        }
        guard()?;
        let hot = collect_required(&allowed, &required, guard)?;
        if !fits_refs(&hot, limits, guard)? {
            return Err(HarnessError::ResourceLimit(
                "mandatory hot tool scope exceeds discovery budget".into(),
            ));
        }
        let (index, cache_hit) = self.catalog_index_for_scope(&allowed, caller, guard)?;
        base_stats.cache_hit = cache_hit;
        let selectable_deferred = limits.max_expansion.min(limits.max_tools) as usize;
        let ranked = rank(&index, query, selectable_deferred, guard)?;
        let exact = matches!(ranked, RankedSelection::Exact(_));
        if exact {
            for (position, id) in ranked.ids().iter().enumerate() {
                if position % DISCOVERY_GUARD_INTERVAL == 0 {
                    guard()?;
                }
                required.insert(id.clone());
            }
            guard()?;
            let exact_entries = collect_required(&allowed, &required, guard)?;
            if !fits_refs(&exact_entries, limits, guard)? {
                return Err(HarnessError::ResourceLimit(
                    "mandatory hot or exact-match tool scope exceeds discovery budget".into(),
                ));
            }
        } else {
            for (ranked_position, id) in ranked.ids().iter().enumerate() {
                if ranked_position % DISCOVERY_GUARD_INTERVAL == 0 {
                    guard()?;
                }
                let mut found = None;
                for (position, entry) in allowed.iter().enumerate() {
                    if position % DISCOVERY_GUARD_INTERVAL == 0 {
                        guard()?;
                    }
                    if entry.definition.id == *id {
                        found = Some(entry);
                        break;
                    }
                }
                let Some(entry) = found else {
                    continue;
                };
                guard()?;
                let mut expanded = required.clone();
                guard()?;
                expanded.insert(entry.definition.id.clone());
                let expanded_entries = collect_required(&allowed, &expanded, guard)?;
                if fits_refs(&expanded_entries, limits, guard)? {
                    required = expanded;
                }
            }
        }
        guard()?;
        let selected = collect_required(&allowed, &required, guard)?;
        scope_with_stats(self, caller, selected, base_stats, true, guard)
    }
}

fn collect_all<'a>(
    entries: &'a [AllowedCatalogEntry<'a>],
    guard: &mut impl FnMut() -> Result<(), HarnessError>,
) -> Result<Vec<&'a AllowedCatalogEntry<'a>>, HarnessError> {
    let mut selected = Vec::with_capacity(entries.len());
    for (position, entry) in entries.iter().enumerate() {
        if position % DISCOVERY_GUARD_INTERVAL == 0 {
            guard()?;
        }
        selected.push(entry);
    }
    guard()?;
    Ok(selected)
}

fn collect_required<'a>(
    entries: &'a [AllowedCatalogEntry<'a>],
    required: &BTreeSet<String>,
    guard: &mut impl FnMut() -> Result<(), HarnessError>,
) -> Result<Vec<&'a AllowedCatalogEntry<'a>>, HarnessError> {
    let mut selected = Vec::new();
    for (position, entry) in entries.iter().enumerate() {
        if position % DISCOVERY_GUARD_INTERVAL == 0 {
            guard()?;
        }
        if required.contains(&entry.definition.id) {
            selected.push(entry);
        }
    }
    guard()?;
    Ok(selected)
}

fn scope_with_stats(
    registry: &ToolRegistry,
    caller: ToolCaller,
    mut entries: Vec<&AllowedCatalogEntry<'_>>,
    mut stats: ToolDiscoveryStats,
    exceeded: bool,
    guard: &mut impl FnMut() -> Result<(), HarnessError>,
) -> Result<(ToolScope, ToolDiscoveryStats), HarnessError> {
    if entries.is_empty() {
        stats.catalog_exceeded_budget = exceeded;
        return Ok((ToolScope::empty(caller), stats));
    }
    if exceeded {
        guard()?;
        entries.sort_by(|left, right| left.definition.id.cmp(&right.definition.id));
        guard()?;
    }
    let cache_key = PreparedCatalogKey::new(&entries, guard)?;
    guard()?;
    #[cfg(test)]
    if let Some(checkpoint) = &registry.prepared_cache_read_checkpoint {
        checkpoint(caller);
    }
    let cached = {
        let cache = registry
            .prepared_catalog_cache
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        cache.get(&cache_key)
    };
    guard()?;
    if let Some(prepared) = cached {
        stats.selected_count = prepared.definitions().len() as u32;
        stats.catalog_exceeded_budget = exceeded;
        let mut tool_ids = BTreeSet::new();
        for (position, definition) in prepared.definitions().iter().enumerate() {
            if position % DISCOVERY_GUARD_INTERVAL == 0 {
                guard()?;
            }
            tool_ids.insert(definition.id.clone());
        }
        guard()?;
        return Ok((
            ToolScope {
                caller,
                tool_ids,
                prepared: Some(prepared),
            },
            stats,
        ));
    }
    let mut definitions = Vec::with_capacity(entries.len());
    let mut tool_ids = BTreeSet::new();
    for (position, entry) in entries.iter().enumerate() {
        if position % DISCOVERY_GUARD_INTERVAL == 0 {
            guard()?;
        }
        definitions.push(entry.definition.clone());
        tool_ids.insert(entry.definition.id.clone());
    }
    guard()?;
    let serialized_definitions = serialize_definition_array(&entries, guard)?;
    let provider_tools = serialize_provider_tool_array(&entries, guard)?;
    let provider_tools = serde_json::value::RawValue::from_string(
        String::from_utf8(provider_tools.to_vec()).map_err(|error| {
            HarnessError::InvalidTool(format!("prepared tool JSON is not UTF-8: {error}"))
        })?,
    )
    .map_err(|error| {
        HarnessError::InvalidTool(format!("prepared tool JSON is invalid: {error}"))
    })?;
    stats.selected_count = definitions.len() as u32;
    let prepared = Arc::new(PreparedToolCatalog::new(
        Arc::from(definitions),
        serialized_definitions,
        provider_tools,
    ));
    stats.catalog_exceeded_budget = exceeded;
    guard()?;
    let mut cache = registry
        .prepared_catalog_cache
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    guard()?;
    let prepared = if let Some(existing) = cache.get(&cache_key) {
        existing
    } else {
        if cache.insert(cache_key, Arc::clone(&prepared)) {
            registry
                .prepared_catalog_build_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        prepared
    };
    drop(cache);
    guard()?;
    Ok((
        ToolScope {
            caller,
            tool_ids,
            prepared: Some(prepared),
        },
        stats,
    ))
}

fn serialize_provider_tool_array(
    entries: &[&AllowedCatalogEntry<'_>],
    guard: &mut impl FnMut() -> Result<(), HarnessError>,
) -> Result<Arc<[u8]>, HarnessError> {
    serialize_fragment_array(
        entries
            .iter()
            .map(|entry| entry.serialized_provider_tool.as_ref()),
        guard,
    )
}

fn serialize_fragment_array<'a>(
    fragments: impl Iterator<Item = &'a [u8]>,
    guard: &mut impl FnMut() -> Result<(), HarnessError>,
) -> Result<Arc<[u8]>, HarnessError> {
    let mut serialized = Vec::new();
    serialized.push(b'[');
    for (position, fragment) in fragments.enumerate() {
        if position % DISCOVERY_GUARD_INTERVAL == 0 {
            guard()?;
        }
        if position > 0 {
            serialized.push(b',');
        }
        serialized.extend_from_slice(fragment);
    }
    serialized.push(b']');
    guard()?;
    Ok(Arc::from(serialized))
}

fn serialize_definition_array(
    entries: &[&AllowedCatalogEntry<'_>],
    guard: &mut impl FnMut() -> Result<(), HarnessError>,
) -> Result<Arc<[u8]>, HarnessError> {
    let mut capacity = entries.len().saturating_sub(1).saturating_add(2);
    for (position, entry) in entries.iter().enumerate() {
        if position % DISCOVERY_GUARD_INTERVAL == 0 {
            guard()?;
        }
        capacity = capacity.saturating_add(entry.serialized_definition.len());
    }
    guard()?;
    let mut serialized = Vec::with_capacity(capacity);
    serialized.push(b'[');
    for (position, entry) in entries.iter().enumerate() {
        if position % DISCOVERY_GUARD_INTERVAL == 0 {
            guard()?;
        }
        if position > 0 {
            serialized.push(b',');
        }
        serialized.extend_from_slice(entry.serialized_definition);
    }
    serialized.push(b']');
    guard()?;
    Ok(Arc::from(serialized))
}

fn fits(
    entries: &[AllowedCatalogEntry<'_>],
    limits: EffectiveLimits,
    guard: &mut impl FnMut() -> Result<(), HarnessError>,
) -> Result<bool, HarnessError> {
    fits_lengths(
        entries.len(),
        entries
            .iter()
            .map(|entry| entry.serialized_definition.len() as u64),
        limits,
        guard,
    )
}

fn fits_refs(
    entries: &[&AllowedCatalogEntry<'_>],
    limits: EffectiveLimits,
    guard: &mut impl FnMut() -> Result<(), HarnessError>,
) -> Result<bool, HarnessError> {
    fits_lengths(
        entries.len(),
        entries
            .iter()
            .map(|entry| entry.serialized_definition.len() as u64),
        limits,
        guard,
    )
}

fn fits_lengths(
    count: usize,
    mut lengths: impl Iterator<Item = u64>,
    limits: EffectiveLimits,
    guard: &mut impl FnMut() -> Result<(), HarnessError>,
) -> Result<bool, HarnessError> {
    let mut elements = Some(0u64);
    for (position, length) in lengths.by_ref().enumerate() {
        if position % DISCOVERY_GUARD_INTERVAL == 0 {
            guard()?;
        }
        elements = elements.and_then(|total| total.checked_add(length));
    }
    guard()?;
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

#[derive(Default)]
struct RankOperationCounts {
    logarithm_evaluations: usize,
    scored_documents: usize,
    retained_candidates_high_water: usize,
}

fn rank(
    index: &CatalogIndex,
    query: &str,
    max_expansion: usize,
    guard: &mut impl FnMut() -> Result<(), HarnessError>,
) -> Result<RankedSelection, HarnessError> {
    rank_with_counts(
        index,
        query,
        max_expansion,
        guard,
        &mut RankOperationCounts::default(),
    )
}

fn rank_with_counts(
    index: &CatalogIndex,
    query: &str,
    max_expansion: usize,
    guard: &mut impl FnMut() -> Result<(), HarnessError>,
    counts: &mut RankOperationCounts,
) -> Result<RankedSelection, HarnessError> {
    let entries = &index.entries;
    let normalized = normalize_phrase(query);
    let query_terms = tokenize(query).into_iter().collect::<BTreeSet<_>>();
    guard()?;
    let exact_id = collect_exact_ids(entries, |entry| entry.id == normalized, guard)?;
    if exact_id.len() == 1 {
        guard()?;
        return Ok(RankedSelection::Exact(exact_id));
    }
    guard()?;
    let exact_name = collect_exact_ids(
        entries,
        |entry| normalize_phrase(&entry.name) == normalized,
        guard,
    )?;
    if exact_name.len() == 1 {
        guard()?;
        return Ok(RankedSelection::Exact(exact_name));
    }
    if exact_name.len() > 1 {
        guard()?;
        return Ok(RankedSelection::Exact(Vec::new()));
    }
    guard()?;
    let exact_namespace = collect_exact_ids(
        entries,
        |entry| entry.metadata.namespace.as_deref() == Some(normalized.as_str()),
        guard,
    )?;
    if !exact_namespace.is_empty() {
        guard()?;
        return Ok(RankedSelection::Exact(exact_namespace));
    }
    guard()?;
    let exact_alias = collect_exact_ids(
        entries,
        |entry| {
            entry
                .metadata
                .aliases
                .iter()
                .any(|alias| alias == &normalized)
        },
        guard,
    )?;
    if exact_alias.len() == 1 {
        guard()?;
        return Ok(RankedSelection::Exact(exact_alias));
    }
    if exact_alias.len() > 1 {
        guard()?;
        return Ok(RankedSelection::Exact(Vec::new()));
    }
    if max_expansion == 0 {
        guard()?;
        return Ok(RankedSelection::Lexical(Vec::new()));
    }
    if query_terms.is_empty() {
        guard()?;
        return Ok(RankedSelection::Lexical(Vec::new()));
    }
    let documents = entries.len() as u128;
    if index.total_document_len == 0 {
        guard()?;
        return Ok(RankedSelection::Lexical(Vec::new()));
    }
    let mut document_scores = vec![0u64; entries.len()];
    let mut matching_postings = 0usize;
    for term in query_terms {
        let Some(postings) = index.postings.get(&term) else {
            continue;
        };
        let inverse_document_frequency = bm25_inverse_document_frequency(
            documents,
            u128::try_from(postings.len()).unwrap_or(u128::MAX),
        );
        counts.logarithm_evaluations = counts.logarithm_evaluations.saturating_add(1);
        for (document, term_weight) in postings {
            if matching_postings.is_multiple_of(DISCOVERY_GUARD_INTERVAL) {
                guard()?;
            }
            matching_postings = matching_postings.saturating_add(1);
            let score = bm25_score_with_idf(inverse_document_frequency, *term_weight);
            document_scores[*document] = document_scores[*document].saturating_add(score);
        }
    }
    guard()?;
    let retained_limit = max_expansion.max(2).min(entries.len());
    let mut scored: Vec<(u64, String)> = Vec::with_capacity(retained_limit);
    for (position, (entry, score)) in entries.iter().zip(document_scores).enumerate() {
        if position % DISCOVERY_GUARD_INTERVAL == 0 {
            guard()?;
        }
        if score > 0 {
            counts.scored_documents = counts.scored_documents.saturating_add(1);
            let candidate = (score, entry.id.clone());
            let insertion = scored
                .binary_search_by(|existing| {
                    existing
                        .0
                        .cmp(&candidate.0)
                        .reverse()
                        .then_with(|| existing.1.cmp(&candidate.1))
                })
                .unwrap_or_else(|position| position);
            if insertion < retained_limit {
                scored.insert(insertion, candidate);
                if scored.len() > retained_limit {
                    scored.pop();
                }
                counts.retained_candidates_high_water =
                    counts.retained_candidates_high_water.max(scored.len());
            }
        }
    }
    guard()?;
    let Some((best, _)) = scored.first() else {
        return Ok(RankedSelection::Lexical(Vec::new()));
    };
    let confident = scored
        .get(1)
        .is_none_or(|(second, _)| best.saturating_sub(*second) >= MIN_CONFIDENCE_MARGIN);
    let take = if confident { 1 } else { max_expansion };
    Ok(RankedSelection::Lexical(
        scored.into_iter().take(take).map(|(_, id)| id).collect(),
    ))
}

fn collect_exact_ids(
    entries: &[CatalogEntry],
    mut predicate: impl FnMut(&CatalogEntry) -> bool,
    guard: &mut impl FnMut() -> Result<(), HarnessError>,
) -> Result<Vec<String>, HarnessError> {
    let mut matches = Vec::new();
    for (position, entry) in entries.iter().enumerate() {
        if position % DISCOVERY_GUARD_INTERVAL == 0 {
            guard()?;
        }
        if predicate(entry) {
            matches.push(entry.id.clone());
        }
    }
    guard()?;
    Ok(matches)
}

#[cfg(test)]
fn bm25_term_score(
    documents: u128,
    document_frequency: u128,
    term_frequency: u128,
    document_len: u128,
    total_document_len: u128,
) -> u64 {
    if documents == 0 || document_frequency == 0 || term_frequency == 0 || total_document_len == 0 {
        return 0;
    }
    let inverse_document_frequency = bm25_inverse_document_frequency(documents, document_frequency);
    let length_normalization =
        bm25_length_normalization(documents, document_len, total_document_len);
    let term_weight = bm25_term_weight(term_frequency, length_normalization);
    bm25_score_with_idf(inverse_document_frequency, term_weight)
}

fn bm25_inverse_document_frequency(documents: u128, document_frequency: u128) -> u128 {
    if documents == 0 || document_frequency == 0 {
        return 0;
    }
    let document_frequency = document_frequency.min(documents);
    let idf_numerator = documents.saturating_add(1).saturating_mul(2);
    let idf_denominator = document_frequency.saturating_mul(2).saturating_add(1);
    scaled_natural_log_ratio(idf_numerator, idf_denominator)
}

fn bm25_length_normalization(
    documents: u128,
    document_len: u128,
    total_document_len: u128,
) -> u128 {
    if documents == 0 || total_document_len == 0 {
        return 0;
    }
    BM25_SCALE
        .saturating_sub(BM25_B)
        .saturating_add(rounded_divide(
            BM25_B
                .saturating_mul(document_len)
                .saturating_mul(documents),
            total_document_len,
        ))
}

fn bm25_term_weight(term_frequency: u128, length_normalization: u128) -> u128 {
    if term_frequency == 0 || length_normalization == 0 {
        return 0;
    }
    let denominator = term_frequency
        .saturating_mul(BM25_SCALE)
        .saturating_add(rounded_divide(
            BM25_K1.saturating_mul(length_normalization),
            BM25_SCALE,
        ));
    rounded_divide(
        term_frequency
            .saturating_mul(BM25_K1.saturating_add(BM25_SCALE))
            .saturating_mul(BM25_SCALE),
        denominator,
    )
}

fn bm25_score_with_idf(inverse_document_frequency: u128, term_weight: u128) -> u64 {
    if inverse_document_frequency == 0 || term_weight == 0 {
        return 0;
    }
    let score = rounded_divide(
        inverse_document_frequency.saturating_mul(term_weight),
        BM25_SCALE,
    );
    u64::try_from(score).unwrap_or(u64::MAX)
}

fn scaled_natural_log_ratio(numerator: u128, denominator: u128) -> u128 {
    if denominator == 0 || numerator <= denominator {
        return 0;
    }
    let mut reduced_denominator = denominator;
    let mut powers_of_two = 0u32;
    while reduced_denominator <= numerator / 2 {
        reduced_denominator = reduced_denominator.saturating_mul(2);
        powers_of_two = powers_of_two.saturating_add(1);
    }

    // z = (x - 1) / (x + 1), with x in [1, 2), so z is in [0, 1/3).
    let z = rounded_divide(
        numerator
            .saturating_sub(reduced_denominator)
            .saturating_mul(BM25_SCALE),
        numerator.saturating_add(reduced_denominator),
    );
    let z_squared = rounded_divide(z.saturating_mul(z), BM25_SCALE);
    let mut power = z;
    let mut series = 0u128;
    for index in 0..LN_SERIES_TERMS {
        series = series.saturating_add(rounded_divide(power, u128::from(index * 2 + 1)));
        power = rounded_divide(power.saturating_mul(z_squared), BM25_SCALE);
    }
    u128::from(powers_of_two)
        .saturating_mul(LN_2_SCALED)
        .saturating_add(series.saturating_mul(2))
}

fn rounded_divide(numerator: u128, denominator: u128) -> u128 {
    if denominator == 0 {
        return u128::MAX;
    }
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    quotient.saturating_add(u128::from(remainder >= denominator / 2 + denominator % 2))
}

fn indexed_terms(entry: &CatalogEntry) -> BTreeMap<String, u32> {
    let mut terms = BTreeMap::<String, u32>::new();
    let mut term_count = 0usize;
    let mut lexical_bytes = 0usize;
    'values: for value in std::iter::once(entry.id.as_str())
        .chain(std::iter::once(entry.name.as_str()))
        .chain(entry.metadata.namespace.iter().map(String::as_str))
        .chain(entry.metadata.aliases.iter().map(String::as_str))
    {
        for term in tokenize(value) {
            let next_count = term_count.saturating_add(1);
            let next_bytes = lexical_bytes.saturating_add(term.len());
            if next_count > MAX_DEFERRED_LEXICAL_TERMS || next_bytes > MAX_DEFERRED_LEXICAL_BYTES {
                break 'values;
            }
            term_count = next_count;
            lexical_bytes = next_bytes;
            let frequency = terms.entry(term).or_default();
            *frequency = frequency.saturating_add(1);
        }
    }
    terms
}

fn tokenize(value: &str) -> Vec<String> {
    bounded_index_value(value)
        .chars()
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

pub(crate) fn bounded_index_value(value: &str) -> String {
    value
        .char_indices()
        .take_while(|(index, _)| *index < MAX_DISCOVERY_QUERY_BYTES)
        .map(|(_, character)| character)
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
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Barrier,
    };
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
    fn synchronized_large_catalog_stop_never_publishes_a_partial_cache() {
        let count = 1_000usize;
        for timed_out in [false, true] {
            let mut registry = ToolRegistry::default();
            for id in allowlist(count) {
                register(
                    &mut registry,
                    &id,
                    "description",
                    ToolDiscoveryMetadata::deferred(),
                );
            }
            let synchronized = Arc::new(Barrier::new(2));
            let stopped = Arc::new(AtomicBool::new(false));
            let worker_barrier = Arc::clone(&synchronized);
            let worker_stopped = Arc::clone(&stopped);
            let worker = std::thread::spawn(move || {
                worker_barrier.wait();
                worker_stopped.store(true, Ordering::Release);
            });
            let minimum_traversal_checkpoints = count.div_ceil(DISCOVERY_GUARD_INTERVAL);
            let mut checkpoints = 0usize;
            let mut synchronized_once = false;
            let result = registry.select_scope_guarded(
                "tool 0999",
                &allowlist(count),
                ToolCaller::Direct,
                ToolDiscoveryLimits::new().with_max_tools(1),
                &ProviderCapabilityLimits::new(),
                &mut || {
                    checkpoints = checkpoints.saturating_add(1);
                    if !synchronized_once && checkpoints >= minimum_traversal_checkpoints {
                        synchronized_once = true;
                        synchronized.wait();
                        while !stopped.load(Ordering::Acquire) {
                            std::thread::yield_now();
                        }
                    }
                    if stopped.load(Ordering::Acquire) {
                        if timed_out {
                            Err(HarnessError::TimedOut("synchronized deadline".into()))
                        } else {
                            Err(HarnessError::Cancelled)
                        }
                    } else {
                        Ok(())
                    }
                },
            );
            worker.join().unwrap();
            assert!(synchronized_once);
            assert!(checkpoints >= minimum_traversal_checkpoints);
            if timed_out {
                assert!(matches!(result, Err(HarnessError::TimedOut(_))));
            } else {
                assert!(matches!(result, Err(HarnessError::Cancelled)));
            }
            assert!(registry.catalog_cache_is_empty());
            assert_eq!(registry.catalog_build_count(), 0);
            assert_eq!(registry.catalog_cache_metrics(), (0, 0, 0));
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
    fn fixed_point_bm25_saturates_tf_and_normalizes_document_length() {
        let tf_one = bm25_term_score(4, 2, 1, 10, 40);
        let tf_two = bm25_term_score(4, 2, 2, 10, 40);
        let tf_four = bm25_term_score(4, 2, 4, 10, 40);
        let tf_eight = bm25_term_score(4, 2, 8, 10, 40);
        assert!(tf_one < tf_two && tf_two < tf_four && tf_four < tf_eight);
        assert!(tf_two - tf_one > tf_eight - tf_four);

        let short = bm25_term_score(4, 2, 2, 4, 40);
        let long = bm25_term_score(4, 2, 2, 20, 40);
        assert!(short > long);
        assert!(bm25_term_score(u128::MAX, 1, u128::MAX, 1, u128::MAX) > 0);
    }

    #[test]
    fn fixed_point_bm25_matches_standard_f64_golden_references() {
        fn reference(
            documents: u128,
            document_frequency: u128,
            term_frequency: u128,
            document_len: u128,
            total_document_len: u128,
        ) -> f64 {
            let documents = documents as f64;
            let document_frequency = document_frequency as f64;
            let term_frequency = term_frequency as f64;
            let document_len = document_len as f64;
            let average_document_len = total_document_len as f64 / documents;
            let idf =
                (1.0 + (documents - document_frequency + 0.5) / (document_frequency + 0.5)).ln();
            idf * term_frequency * (1.2 + 1.0)
                / (term_frequency + 1.2 * (1.0 - 0.75 + 0.75 * document_len / average_document_len))
        }

        let cases = [
            (1, 1, 1, 1, 1),
            (4, 2, 1, 10, 40),
            (10, 1, 3, 4, 80),
            (100, 90, 2, 40, 2_500),
            (1_000, 1, 32, 250, 100_000),
        ];
        for (documents, df, tf, document_len, total_document_len) in cases {
            let fixed = bm25_term_score(documents, df, tf, document_len, total_document_len);
            let actual = fixed as f64 / BM25_SCALE as f64;
            let expected = reference(documents, df, tf, document_len, total_document_len);
            assert!(
                (actual - expected).abs() <= 5e-8,
                "N={documents} df={df} tf={tf} dl={document_len}: {actual} != {expected}"
            );
        }

        assert_eq!(scaled_natural_log_ratio(10, 5), LN_2_SCALED);
        assert_eq!(scaled_natural_log_ratio(4, 3), 287_682_072);
        let multi_term =
            bm25_term_score(10, 1, 1, 4, 80).saturating_add(bm25_term_score(10, 5, 2, 4, 80));
        let reversed =
            bm25_term_score(10, 5, 2, 4, 80).saturating_add(bm25_term_score(10, 1, 1, 4, 80));
        assert_eq!(multi_term, reversed);
        assert!(multi_term > bm25_term_score(10, 1, 1, 4, 80));

        let saturated = bm25_term_score(u128::MAX, 1, u128::MAX, 1, u128::MAX);
        assert_eq!(
            saturated,
            bm25_term_score(u128::MAX, 1, u128::MAX, 1, u128::MAX)
        );
        assert!(saturated > 0);
    }

    #[test]
    fn bm25_ranking_is_length_sensitive_and_permutation_stable() {
        fn entry(id: &str, name: &str) -> CatalogEntry {
            CatalogEntry {
                id: id.into(),
                name: name.into(),
                metadata: ToolDiscoveryMetadata::deferred(),
                terms: BTreeMap::new(),
                document_len: 0,
            }
        }

        let entries = vec![
            entry("rank.a", "needle"),
            entry(
                "rank.b",
                "needle filler filler filler filler filler filler filler",
            ),
            entry("rank.c", "unrelated"),
        ];
        let mut reversed = entries.clone();
        reversed.reverse();
        for candidate in [entries, reversed] {
            let index = CatalogIndex::build(candidate, &mut || Ok(())).unwrap();
            let ranked = rank(&index, "needle request", 1, &mut || Ok(())).unwrap();
            assert_eq!(ranked.ids(), &["rank.a"]);
        }

        let tied = CatalogIndex::build(
            vec![entry("tie.b", "needle"), entry("tie.a", "needle")],
            &mut || Ok(()),
        )
        .unwrap();
        let ranked = rank(&tied, "needle request", 1, &mut || Ok(())).unwrap();
        assert_eq!(ranked.ids(), &["tie.a"]);
    }

    #[test]
    fn adversarial_rank_precomputes_idf_and_retains_only_bounded_top_candidates() {
        let terms = (0..MAX_QUERY_TERMS)
            .map(|index| format!("term{index:02}"))
            .collect::<Vec<_>>();
        let document = terms.join(" ");
        let entries = (0..1_000)
            .map(|index| CatalogEntry {
                id: format!("adversarial.tool.{index:04}"),
                name: document.clone(),
                metadata: ToolDiscoveryMetadata::deferred(),
                terms: BTreeMap::new(),
                document_len: 0,
            })
            .collect::<Vec<_>>();
        let index = CatalogIndex::build(entries, &mut || Ok(())).unwrap();
        let mut counts = RankOperationCounts::default();
        let query = format!("{document} request");
        let ranked = rank_with_counts(&index, &query, 8, &mut || Ok(()), &mut counts).unwrap();

        assert_eq!(counts.logarithm_evaluations, MAX_QUERY_TERMS);
        assert!(counts.logarithm_evaluations <= tokenize(&query).len());
        assert_eq!(counts.scored_documents, 1_000);
        assert_eq!(counts.retained_candidates_high_water, 8);
        assert_eq!(ranked.ids().len(), 8);
        assert_eq!(
            ranked.ids().first().map(String::as_str),
            Some("adversarial.tool.0000")
        );
        assert_eq!(
            ranked.ids().last().map(String::as_str),
            Some("adversarial.tool.0007")
        );
    }

    #[test]
    fn exact_scan_checks_stop_again_immediately_before_returning() {
        let entries = (0..100)
            .map(|index| CatalogEntry {
                id: format!("exact.tool.{index:03}"),
                name: format!("Exact tool {index:03}"),
                metadata: ToolDiscoveryMetadata::deferred(),
                terms: BTreeMap::new(),
                document_len: 0,
            })
            .collect::<Vec<_>>();
        let index = CatalogIndex::build(entries, &mut || Ok(())).unwrap();
        let stop_before_exact_return =
            1 + index.entries.len().div_ceil(DISCOVERY_GUARD_INTERVAL) + 2;
        let mut checkpoints = 0usize;
        let result = rank(&index, "exact.tool.099", 1, &mut || {
            checkpoints = checkpoints.saturating_add(1);
            if checkpoints >= stop_before_exact_return {
                Err(HarnessError::Cancelled)
            } else {
                Ok(())
            }
        });
        assert!(matches!(result, Err(HarnessError::Cancelled)));
        assert!(checkpoints >= stop_before_exact_return);
    }

    #[test]
    fn successful_scope_output_is_independent_of_checkpoint_scheduling() {
        fn registry() -> ToolRegistry {
            let mut registry = ToolRegistry::default();
            for index in 0..100 {
                register(
                    &mut registry,
                    &format!("timing.tool.{index:03}"),
                    "description",
                    ToolDiscoveryMetadata::deferred(),
                );
            }
            registry
        }

        let allowlist = (0..100)
            .map(|index| format!("timing.tool.{index:03}"))
            .collect::<Vec<_>>();
        let select = |registry: &ToolRegistry, cadence: usize| {
            let mut checkpoints = 0usize;
            let selected = registry
                .select_scope_guarded(
                    "timing.tool.099",
                    &allowlist,
                    ToolCaller::Direct,
                    ToolDiscoveryLimits::new().with_max_tools(1),
                    &ProviderCapabilityLimits::new(),
                    &mut || {
                        checkpoints = checkpoints.saturating_add(1);
                        if checkpoints.is_multiple_of(cadence) {
                            std::thread::yield_now();
                        }
                        Ok(())
                    },
                )
                .unwrap();
            (selected, checkpoints)
        };

        let ((baseline, baseline_stats), baseline_checkpoints) = select(&registry(), usize::MAX);
        for cadence in [1, 2, 7] {
            let ((selected, stats), checkpoints) = select(&registry(), cadence);
            assert_eq!(selected.definitions(), baseline.definitions());
            assert_eq!(
                selected.serialized_definitions(),
                baseline.serialized_definitions()
            );
            assert_eq!(stats.candidate_count, baseline_stats.candidate_count);
            assert_eq!(stats.selected_count, baseline_stats.selected_count);
            assert_eq!(
                stats.deferred_candidate_count,
                baseline_stats.deferred_candidate_count
            );
            assert_eq!(
                stats.catalog_exceeded_budget,
                baseline_stats.catalog_exceeded_budget
            );
            assert!(checkpoints >= allowlist.len().div_ceil(DISCOVERY_GUARD_INTERVAL));
        }
        assert!(baseline_checkpoints >= allowlist.len().div_ceil(DISCOVERY_GUARD_INTERVAL));
    }

    #[test]
    fn serialized_fragments_and_scope_cache_are_exact_and_caller_isolated() {
        let mut registry = ToolRegistry::default();
        for (id, name, alias) in [
            ("fragment.one", r#"One, [quoted] "name""#, "needle-one"),
            ("fragment.two", r#"Two }{ backslash \"#, "needle-two"),
            ("fragment.three", "Three: delimiter", "needle-three"),
        ] {
            let definition = ToolDefinition::new(
                id,
                name,
                "description with ],[{ delimiters",
                json!({"type": "object", "properties": {"value": {"type": "string"}}}),
            )
            .with_allowed_callers([ToolCaller::Direct, ToolCaller::DeclarativePlan]);
            registry
                .register_with_discovery(
                    Arc::new(TestTool(definition)),
                    ToolDiscoveryMetadata::deferred().with_aliases([alias]),
                )
                .unwrap();
        }
        let ids = vec![
            "fragment.one".into(),
            "fragment.two".into(),
            "fragment.three".into(),
        ];
        let limits = ToolDiscoveryLimits::new()
            .with_max_tools(2)
            .with_max_expansion_tools(2);
        let (cold, cold_stats) = registry
            .select_scope(
                "needle request",
                &ids,
                ToolCaller::Direct,
                limits,
                &ProviderCapabilityLimits::new(),
            )
            .unwrap();
        assert!(!cold_stats.cache_hit);
        assert_eq!(registry.catalog_build_count(), 1);
        assert_eq!(
            cold.serialized_definitions(),
            serde_json::to_vec(cold.definitions()).unwrap()
        );

        let mut permuted = ids.clone();
        permuted.reverse();
        let (warm, warm_stats) = registry
            .select_scope(
                "needle request",
                &permuted,
                ToolCaller::Direct,
                limits,
                &ProviderCapabilityLimits::new(),
            )
            .unwrap();
        assert!(warm_stats.cache_hit);
        assert_eq!(registry.catalog_build_count(), 1);
        assert_eq!(registry.prepared_catalog_build_count(), 1);
        assert_eq!(warm.definitions(), cold.definitions());
        assert_eq!(
            warm.serialized_definitions(),
            serde_json::to_vec(warm.definitions()).unwrap()
        );

        let (_, declarative_stats) = registry
            .select_scope(
                "needle request",
                &ids,
                ToolCaller::DeclarativePlan,
                limits,
                &ProviderCapabilityLimits::new(),
            )
            .unwrap();
        assert!(!declarative_stats.cache_hit);
        assert_eq!(registry.catalog_build_count(), 2);
        let (_, direct_again) = registry
            .select_scope(
                "needle request",
                &ids,
                ToolCaller::Direct,
                limits,
                &ProviderCapabilityLimits::new(),
            )
            .unwrap();
        assert!(direct_again.cache_hit);
        assert_eq!(registry.catalog_build_count(), 2);
    }

    #[test]
    fn unrelated_registration_cannot_make_an_authorized_scope_cold() {
        let mut registry = ToolRegistry::default();
        for id in ["permitted.alpha", "permitted.beta"] {
            register(
                &mut registry,
                id,
                "description",
                ToolDiscoveryMetadata::deferred(),
            );
        }
        let allowed = vec!["permitted.alpha".into(), "permitted.beta".into()];
        let limits = ToolDiscoveryLimits::new().with_max_tools(1);
        let select = |registry: &ToolRegistry, caller, allowlist: &[String]| {
            registry
                .select_scope(
                    "permitted request",
                    allowlist,
                    caller,
                    limits,
                    &ProviderCapabilityLimits::new(),
                )
                .unwrap()
        };

        let (direct, direct_cold) = select(&registry, ToolCaller::Direct, &allowed);
        let (declarative, declarative_cold) =
            select(&registry, ToolCaller::DeclarativePlan, &allowed);
        assert!(!direct_cold.cache_hit);
        assert!(!declarative_cold.cache_hit);
        assert_eq!(registry.catalog_build_count(), 2);

        register(
            &mut registry,
            "hidden.unallowlisted",
            "description",
            ToolDiscoveryMetadata::deferred(),
        );
        let (direct_after_hidden, direct_warm) = select(&registry, ToolCaller::Direct, &allowed);
        let (declarative_after_hidden, declarative_warm) =
            select(&registry, ToolCaller::DeclarativePlan, &allowed);
        assert!(direct_warm.cache_hit);
        assert!(declarative_warm.cache_hit);
        assert_eq!(direct.definitions(), direct_after_hidden.definitions());
        assert_eq!(
            declarative.definitions(),
            declarative_after_hidden.definitions()
        );
        assert_eq!(registry.catalog_build_count(), 2);

        let incompatible = ToolDefinition::new(
            "hidden.incompatible",
            "Hidden incompatible",
            "description",
            json!({"type": "object"}),
        )
        .with_allowed_callers([ToolCaller::Programmatic]);
        registry
            .register_with_discovery(
                Arc::new(TestTool(incompatible)),
                ToolDiscoveryMetadata::deferred(),
            )
            .unwrap();
        let mut with_incompatible = allowed.clone();
        with_incompatible.push("hidden.incompatible".into());
        for caller in [ToolCaller::Direct, ToolCaller::DeclarativePlan] {
            let (_, stats) = select(&registry, caller, &with_incompatible);
            assert!(stats.cache_hit);
        }
        assert_eq!(registry.catalog_build_count(), 2);

        register(
            &mut registry,
            "permitted.new",
            "description",
            ToolDiscoveryMetadata::deferred(),
        );
        let mut expanded = allowed;
        expanded.push("permitted.new".into());
        for caller in [ToolCaller::Direct, ToolCaller::DeclarativePlan] {
            let (_, stats) = select(&registry, caller, &expanded);
            assert!(!stats.cache_hit);
        }
        assert_eq!(registry.catalog_build_count(), 4);
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
    fn scope_cache_build_work_does_not_traverse_unallowlisted_tools() {
        fn registry_with_hidden(hidden: usize) -> ToolRegistry {
            let mut registry = ToolRegistry::default();
            for index in 0..3 {
                register(
                    &mut registry,
                    &format!("allowed.tool.{index}"),
                    "description",
                    ToolDiscoveryMetadata::deferred()
                        .with_aliases([format!("allowed-alias-{index}")]),
                );
            }
            for index in 0..hidden {
                register(
                    &mut registry,
                    &format!("hidden.tool.{index:04}"),
                    "description",
                    ToolDiscoveryMetadata::deferred()
                        .with_aliases([format!("hidden-alias-{index:04}")]),
                );
            }
            registry
        }

        let ids = vec![
            "allowed.tool.0".into(),
            "allowed.tool.1".into(),
            "allowed.tool.2".into(),
        ];
        let limits = ToolDiscoveryLimits::new().with_max_tools(1);
        let baseline = registry_with_hidden(0);
        let polluted = registry_with_hidden(1_000);
        let guarded_select = |registry: &ToolRegistry| {
            let mut checkpoints = 0u32;
            let result = registry.select_scope_guarded(
                "allowed request",
                &ids,
                ToolCaller::Direct,
                limits,
                &ProviderCapabilityLimits::new(),
                &mut || {
                    checkpoints = checkpoints.saturating_add(1);
                    Ok(())
                },
            );
            (result.unwrap(), checkpoints)
        };
        let ((baseline_scope, baseline_stats), baseline_checkpoints) = guarded_select(&baseline);
        let ((polluted_scope, polluted_stats), polluted_checkpoints) = guarded_select(&polluted);
        assert_eq!(baseline_scope.definitions(), polluted_scope.definitions());
        assert_eq!(
            baseline_stats.candidate_count,
            polluted_stats.candidate_count
        );
        assert_eq!(baseline_checkpoints, polluted_checkpoints);
        assert_eq!(baseline.catalog_build_count(), 1);
        assert_eq!(polluted.catalog_build_count(), 1);
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

        let mut pair_registry = ToolRegistry::default();
        for id in ["pair.one", "pair.two"] {
            register(&mut pair_registry, id, "pair", ToolDiscoveryMetadata::hot());
        }
        let pair_ids = vec!["pair.two".into(), "pair.one".into()];
        let pair_entries = pair_registry.allowed_catalog(&pair_ids, ToolCaller::Direct);
        let pair_bytes = pair_entries
            .iter()
            .map(|entry| entry.serialized_definition.len() as u64)
            .sum::<u64>()
            + 3;
        let pair_exact = ToolDiscoveryLimits::new()
            .with_max_tools(2)
            .with_max_tool_schema_bytes(pair_bytes);
        let (pair, _) = scope(
            &pair_registry,
            "pair",
            &pair_ids,
            pair_exact,
            ProviderCapabilityLimits::new(),
        )
        .unwrap();
        assert_eq!(pair.serialized_definitions().len() as u64, pair_bytes);
        assert_eq!(pair.definitions()[0].id, "pair.two");
        assert!(scope(
            &pair_registry,
            "pair",
            &pair_ids,
            pair_exact.with_max_tool_schema_bytes(pair_bytes - 1),
            ProviderCapabilityLimits::new(),
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
        assert!(empty_registry.catalog_cache_is_empty());
        assert_eq!(empty_registry.catalog_build_count(), 0);

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
        assert!(full_fit_registry.catalog_cache_is_empty());
        assert_eq!(full_fit_registry.catalog_build_count(), 0);
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
    fn deferred_lexical_caps_reject_adversarial_shapes_without_invalidating_warm_caches() {
        let mut registry = ToolRegistry::default();
        for id in ["bounded.alpha", "bounded.beta"] {
            register(
                &mut registry,
                id,
                "description",
                ToolDiscoveryMetadata::deferred(),
            );
        }
        let allowlist = vec!["bounded.alpha".into(), "bounded.beta".into()];
        let limits = ToolDiscoveryLimits::new().with_max_tools(1);
        scope(
            &registry,
            "bounded request",
            &allowlist,
            limits,
            ProviderCapabilityLimits::new(),
        )
        .unwrap();
        let (_, warm) = scope(
            &registry,
            "bounded request",
            &allowlist,
            limits,
            ProviderCapabilityLimits::new(),
        )
        .unwrap();
        assert!(warm.cache_hit);
        let fingerprint = registry.catalog_fingerprint();
        let catalog_metrics = registry.catalog_cache_metrics();
        let prepared_metrics = registry.prepared_catalog_cache_metrics();

        let term_aliases = (0..32)
            .map(|alias| {
                (0..9)
                    .map(|term| format!("t{alias:02}{term}"))
                    .collect::<Vec<_>>()
                    .join("-")
            })
            .collect::<Vec<_>>();
        let term_error = registry
            .register_with_discovery(
                Arc::new(TestTool(definition("deferred.term.shape", "description"))),
                ToolDiscoveryMetadata::deferred().with_aliases(term_aliases),
            )
            .unwrap_err();
        assert!(matches!(
            term_error,
            HarnessError::InvalidTool(message) if message.contains("lexical index limit")
        ));

        let byte_aliases = (0..16)
            .map(|index| format!("a{index:02}{}", "x".repeat(87)))
            .collect::<Vec<_>>();
        let byte_error = registry
            .register_with_discovery(
                Arc::new(TestTool(definition("deferred.byte.shape", "description"))),
                ToolDiscoveryMetadata::deferred().with_aliases(byte_aliases.clone()),
            )
            .unwrap_err();
        assert!(matches!(
            byte_error,
            HarnessError::InvalidTool(message) if message.contains("lexical index limit")
        ));

        assert_eq!(registry.catalog_fingerprint(), fingerprint);
        assert_eq!(registry.catalog_cache_metrics(), catalog_metrics);
        assert_eq!(registry.prepared_catalog_cache_metrics(), prepared_metrics);
        registry
            .register_with_discovery(
                Arc::new(TestTool(definition("hot.byte.shape", "description"))),
                ToolDiscoveryMetadata::hot().with_aliases(byte_aliases),
            )
            .unwrap();
    }

    #[test]
    fn byte_weighted_caches_are_bounded_lru_and_eviction_preserves_output() {
        fn lexical_index() -> Arc<CatalogIndex> {
            let terms = (0..MAX_QUERY_TERMS)
                .map(|index| format!("term{index:02}"))
                .collect::<Vec<_>>()
                .join(" ");
            let entries = (0..1_000)
                .map(|index| CatalogEntry {
                    id: format!("cache.tool.{index:04}"),
                    name: terms.clone(),
                    metadata: ToolDiscoveryMetadata::deferred(),
                    terms: BTreeMap::new(),
                    document_len: 0,
                })
                .collect();
            Arc::new(CatalogIndex::build(entries, &mut || Ok(())).unwrap())
        }

        let index = lexical_index();
        let query = (0..MAX_QUERY_TERMS)
            .map(|term| format!("term{term:02}"))
            .chain(std::iter::once("request".to_owned()))
            .collect::<Vec<_>>()
            .join(" ");
        let expected = rank(&index, &query, 8, &mut || Ok(())).unwrap();
        let keys = (0..MAX_SCOPE_CATALOG_CACHE_ENTRIES)
            .map(|scope| {
                CatalogCacheKey::new(
                    ToolCaller::Direct,
                    (0..1_000)
                        .map(|tool| (format!("scope{scope:02}.tool.{tool:04}"), 1))
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        let entry_weight = keys[0]
            .retained_bytes()
            .saturating_add(index.retained_bytes());
        assert!(entry_weight < MAX_SCOPE_CATALOG_CACHE_BYTES);
        let insertions = MAX_SCOPE_CATALOG_CACHE_BYTES / entry_weight + 1;
        assert!(insertions > 2 && insertions <= MAX_SCOPE_CATALOG_CACHE_ENTRIES);
        let mut cache = CatalogCache::default();
        for key in keys.iter().take(insertions - 1) {
            assert!(cache.insert(key.clone(), Arc::clone(&index)));
        }
        assert!(cache.get(&keys[0]).is_some());
        assert!(cache.insert(keys[insertions - 1].clone(), Arc::clone(&index)));
        assert!(cache.get(&keys[1]).is_none());
        assert!(cache.get(&keys[0]).is_some());
        assert_eq!(cache.evictions(), 1);
        assert!(cache.retained_bytes() <= MAX_SCOPE_CATALOG_CACHE_BYTES);
        assert!(cache.len() < MAX_SCOPE_CATALOG_CACHE_ENTRIES);
        let after_eviction = rank(&index, &query, 8, &mut || Ok(())).unwrap();
        assert_eq!(after_eviction.ids(), expected.ids());

        let description = "x".repeat(1024 * 1024);
        let make_prepared = |index| {
            let id = format!("prepared.{index:02}");
            let definition = ToolDefinition::new(&id, &id, &description, json!({"type": "object"}));
            let catalog =
                Arc::new(PreparedToolCatalog::from_definitions(vec![definition]).unwrap());
            let key = PreparedCatalogKey(Arc::from(vec![(id, 1)]));
            (key, catalog)
        };
        let first_prepared = make_prepared(0);
        let prepared_weight = first_prepared
            .0
            .retained_bytes()
            .saturating_add(prepared_catalog_retained_bytes(&first_prepared.1));
        assert!(prepared_weight < MAX_PREPARED_CATALOG_CACHE_BYTES);
        let prepared_insertions = MAX_PREPARED_CATALOG_CACHE_BYTES / prepared_weight + 1;
        assert!(
            prepared_insertions > 2 && prepared_insertions <= MAX_PREPARED_CATALOG_CACHE_ENTRIES
        );
        let mut prepared = Vec::with_capacity(prepared_insertions);
        prepared.push(first_prepared);
        prepared.extend((1..prepared_insertions).map(make_prepared));
        let first_provider_json = prepared[0].1.provider_tools_json().get().to_owned();
        let mut prepared_cache = PreparedCatalogCache::default();
        for (key, catalog) in prepared.iter().take(prepared_insertions - 1) {
            assert!(prepared_cache.insert(key.clone(), Arc::clone(catalog)));
        }
        assert!(prepared_cache.get(&prepared[0].0).is_some());
        assert!(prepared_cache.insert(
            prepared[prepared_insertions - 1].0.clone(),
            Arc::clone(&prepared[prepared_insertions - 1].1),
        ));
        assert!(prepared_cache.get(&prepared[1].0).is_none());
        assert!(prepared_cache.get(&prepared[0].0).is_some());
        assert_eq!(prepared_cache.evictions(), 1);
        assert!(prepared_cache.retained_bytes() <= MAX_PREPARED_CATALOG_CACHE_BYTES);
        assert!(prepared_cache.len() < MAX_PREPARED_CATALOG_CACHE_ENTRIES);
        assert_eq!(
            prepared[0].1.provider_tools_json().get(),
            first_provider_json
        );

        let tiny_index = Arc::new(
            CatalogIndex::build(
                vec![CatalogEntry {
                    id: "tiny.tool".into(),
                    name: "Tiny tool".into(),
                    metadata: ToolDiscoveryMetadata::deferred(),
                    terms: BTreeMap::new(),
                    document_len: 0,
                }],
                &mut || Ok(()),
            )
            .unwrap(),
        );
        let count_keys = (0..=MAX_SCOPE_CATALOG_CACHE_ENTRIES)
            .map(|index| {
                CatalogCacheKey::new(ToolCaller::Direct, vec![(format!("k{index:02}"), 1)])
            })
            .collect::<Vec<_>>();
        let mut count_cache = CatalogCache::default();
        for key in count_keys.iter().take(MAX_SCOPE_CATALOG_CACHE_ENTRIES) {
            assert!(count_cache.insert(key.clone(), Arc::clone(&tiny_index)));
        }
        assert!(count_cache.get(&count_keys[0]).is_some());
        assert!(count_cache.insert(
            count_keys[MAX_SCOPE_CATALOG_CACHE_ENTRIES].clone(),
            Arc::clone(&tiny_index),
        ));
        assert!(count_cache.get(&count_keys[1]).is_none());
        assert!(count_cache.get(&count_keys[0]).is_some());
        assert_eq!(count_cache.len(), MAX_SCOPE_CATALOG_CACHE_ENTRIES);
        assert_eq!(count_cache.evictions(), 1);
        assert!(count_cache.retained_bytes() <= MAX_SCOPE_CATALOG_CACHE_BYTES);

        let tiny_prepared = Arc::new(
            PreparedToolCatalog::from_definitions(vec![ToolDefinition::new(
                "tiny.prepared",
                "Tiny prepared",
                "description",
                json!({"type": "object"}),
            )])
            .unwrap(),
        );
        let prepared_count_keys = (0..=MAX_PREPARED_CATALOG_CACHE_ENTRIES)
            .map(|index| PreparedCatalogKey(Arc::from(vec![(format!("p{index:02}"), 1)])))
            .collect::<Vec<_>>();
        let mut prepared_count_cache = PreparedCatalogCache::default();
        for key in prepared_count_keys
            .iter()
            .take(MAX_PREPARED_CATALOG_CACHE_ENTRIES)
        {
            assert!(prepared_count_cache.insert(key.clone(), Arc::clone(&tiny_prepared)));
        }
        assert!(prepared_count_cache.get(&prepared_count_keys[0]).is_some());
        assert!(prepared_count_cache.insert(
            prepared_count_keys[MAX_PREPARED_CATALOG_CACHE_ENTRIES].clone(),
            Arc::clone(&tiny_prepared),
        ));
        assert!(prepared_count_cache.get(&prepared_count_keys[1]).is_none());
        assert!(prepared_count_cache.get(&prepared_count_keys[0]).is_some());
        assert_eq!(
            prepared_count_cache.len(),
            MAX_PREPARED_CATALOG_CACHE_ENTRIES
        );
        assert_eq!(prepared_count_cache.evictions(), 1);
        assert!(prepared_count_cache.retained_bytes() <= MAX_PREPARED_CATALOG_CACHE_BYTES);
    }

    #[test]
    fn max_shape_indexes_are_bounded_and_oversized_prepared_entries_are_uncached() {
        let alias_tokens = (0..249)
            .map(|index| format!("{index:02x}"))
            .collect::<Vec<_>>();
        let aliases = alias_tokens
            .chunks(32)
            .map(|chunk| chunk.join("-"))
            .collect::<Vec<_>>();
        let metadata = ToolDiscoveryMetadata::deferred().with_aliases(aliases);
        let mut registry = ToolRegistry::default();
        let mut allowlist = Vec::with_capacity(1_000);
        for index in 0..1_000 {
            let id = format!("maxshape.tool.{index:04}");
            let definition =
                ToolDefinition::new(&id, "needle", "description", json!({"type": "object"}))
                    .with_allowed_callers([ToolCaller::Direct, ToolCaller::DeclarativePlan]);
            registry
                .register_with_discovery(Arc::new(TestTool(definition)), metadata.clone())
                .unwrap();
            allowlist.push(id);
        }
        let limits = ToolDiscoveryLimits::new().with_max_tools(1);
        let (first, first_stats) = scope(
            &registry,
            "unmatched request",
            &allowlist,
            limits,
            ProviderCapabilityLimits::new(),
        )
        .unwrap();
        let (second, second_stats) = scope(
            &registry,
            "unmatched request",
            &allowlist,
            limits,
            ProviderCapabilityLimits::new(),
        )
        .unwrap();
        assert!(first.is_empty() && second.is_empty());
        assert!(!first_stats.cache_hit && second_stats.cache_hit);
        assert_eq!(registry.catalog_build_count(), 1);
        let (entries, bytes, evictions) = registry.catalog_cache_metrics();
        assert_eq!(entries, 1);
        assert!(bytes <= MAX_SCOPE_CATALOG_CACHE_BYTES);
        assert_eq!(evictions, 0);

        let mut prepared_registry = ToolRegistry::default();
        let description = "x".repeat(3 * 1024 * 1024);
        register(
            &mut prepared_registry,
            "oversized.prepared",
            &description,
            ToolDiscoveryMetadata::deferred(),
        );
        let prepared_allowlist = vec!["oversized.prepared".into()];
        let prepared_limits = ToolDiscoveryLimits::new()
            .with_max_tools(1)
            .with_max_tool_schema_bytes(8 * 1024 * 1024);
        let (prepared_first, _) = scope(
            &prepared_registry,
            "oversized.prepared",
            &prepared_allowlist,
            prepared_limits,
            ProviderCapabilityLimits::new(),
        )
        .unwrap();
        let (prepared_second, _) = scope(
            &prepared_registry,
            "oversized.prepared",
            &prepared_allowlist,
            prepared_limits,
            ProviderCapabilityLimits::new(),
        )
        .unwrap();
        assert_eq!(
            prepared_first.serialized_definitions(),
            prepared_second.serialized_definitions()
        );
        assert_eq!(prepared_registry.prepared_catalog_build_count(), 0);
        assert_eq!(
            prepared_registry.prepared_catalog_cache_metrics(),
            (0, 0, 0)
        );
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
        let (_, cold_after_registration) = scope(
            &registry,
            "no match",
            &all_ids,
            ToolDiscoveryLimits::new().with_max_tools(1),
            ProviderCapabilityLimits::new(),
        )
        .unwrap();
        assert!(!cold_after_registration.cache_hit);
        let (_, warmed) = scope(
            &registry,
            "no match",
            &all_ids,
            ToolDiscoveryLimits::new().with_max_tools(1),
            ProviderCapabilityLimits::new(),
        )
        .unwrap();
        assert!(warmed.cache_hit);
        assert_eq!(registry.catalog_build_count(), 1);
        assert!(!registry.catalog_cache_is_empty());

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
        assert_eq!(registry.catalog_build_count(), 1);
        assert!(!registry.catalog_cache_is_empty());
        let (_, preserved) = scope(
            &registry,
            "no match",
            &all_ids,
            ToolDiscoveryLimits::new().with_max_tools(1),
            ProviderCapabilityLimits::new(),
        )
        .unwrap();
        assert!(preserved.cache_hit);
        assert_eq!(registry.catalog_build_count(), 1);
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

    #[test]
    #[ignore = "non-gating release discovery latency evaluation"]
    fn discovery_release_microbenchmark() {
        use std::{hint::black_box, time::Duration};

        fn percentile(samples: &mut [Duration], percentile: usize) -> Duration {
            samples.sort_unstable();
            let position = samples
                .len()
                .saturating_mul(percentile)
                .div_ceil(100)
                .saturating_sub(1);
            samples[position.min(samples.len().saturating_sub(1))]
        }

        fn evaluate(label: &str, count: usize, adversarial: bool) {
            let mut registry = ToolRegistry::default();
            let adversarial_name = (0..MAX_QUERY_TERMS)
                .map(|term| format!("t{term:02}"))
                .collect::<Vec<_>>()
                .join(" ");
            let mut allowlist = Vec::with_capacity(count);
            for index in 0..count {
                let id = format!("benchmark.tool.{index:04}");
                let name = if adversarial {
                    adversarial_name.clone()
                } else {
                    format!("Benchmark tool {index:04}")
                };
                let metadata = if adversarial {
                    ToolDiscoveryMetadata::deferred()
                } else {
                    ToolDiscoveryMetadata::deferred()
                        .with_aliases([format!("category-{:02}", index % 10)])
                };
                let definition = ToolDefinition::new(
                    &id,
                    name,
                    "description excluded from discovery",
                    json!({"type": "object"}),
                )
                .with_allowed_callers([ToolCaller::Direct]);
                registry
                    .register_with_discovery(Arc::new(TestTool(definition)), metadata)
                    .unwrap();
                allowlist.push(id);
            }
            let query = if adversarial {
                format!("{adversarial_name} request")
            } else {
                "category request".to_owned()
            };
            let limits = ToolDiscoveryLimits::new()
                .with_max_tools(4)
                .with_max_expansion_tools(4);
            let provider = ProviderCapabilityLimits::new();

            let mut cold = Vec::with_capacity(25);
            for _ in 0..25 {
                *registry
                    .catalog_cache
                    .write()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = CatalogCache::default();
                *registry
                    .prepared_catalog_cache
                    .write()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) =
                    PreparedCatalogCache::default();
                let provider_limits = provider.clone();
                let started = std::time::Instant::now();
                let (selected, stats) =
                    scope(&registry, &query, &allowlist, limits, provider_limits)
                        .expect("benchmark scope selection");
                cold.push(started.elapsed());
                assert!(!stats.cache_hit);
                black_box(selected.serialized_definitions());
            }

            let (selected, _) = scope(&registry, &query, &allowlist, limits, provider.clone())
                .expect("benchmark warmup");
            black_box(selected.serialized_definitions());
            let mut warm = Vec::with_capacity(200);
            for _ in 0..200 {
                let provider_limits = provider.clone();
                let started = std::time::Instant::now();
                let (selected, stats) =
                    scope(&registry, &query, &allowlist, limits, provider_limits)
                        .expect("benchmark warm scope selection");
                warm.push(started.elapsed());
                assert!(stats.cache_hit);
                black_box(selected.serialized_definitions());
            }

            let cold_median = percentile(&mut cold, 50);
            let cold_p95 = percentile(&mut cold, 95);
            let warm_median = percentile(&mut warm, 50);
            let warm_p95 = percentile(&mut warm, 95);
            println!(
                "discovery-benchmark scenario={label} tools={count} cold_median_us={} cold_p95_us={} warm_median_us={} warm_p95_us={} cold_target={} warm_target={}",
                cold_median.as_micros(),
                cold_p95.as_micros(),
                warm_median.as_micros(),
                warm_p95.as_micros(),
                if cold_p95 < Duration::from_millis(50) { "pass" } else { "miss" },
                if warm_p95 < Duration::from_millis(5) { "pass" } else { "miss" },
            );
        }

        println!(
            "discovery-benchmark profile={}",
            if cfg!(debug_assertions) {
                "debug (results are informational; rerun with --release)"
            } else {
                "release"
            }
        );
        for count in [30, 100, 1_000] {
            evaluate("normal", count, false);
        }
        evaluate("adversarial-64-term", 1_000, true);
    }
}
