[CmdletBinding()]
param(
    [string]$BaselinePath = (Join-Path $PSScriptRoot "..\\..\\.github\\semver\\rust-0.1.0-baseline.json"),
    [switch]$ValidateOnly,
    [switch]$RecordInitialBaseline
)

$ErrorActionPreference = "Stop"
$expectedVersion = "0.1.0"
$expectedTag = "v$expectedVersion"

if ($ValidateOnly -and $RecordInitialBaseline) {
    throw "-ValidateOnly and -RecordInitialBaseline cannot be used together."
}

if (-not (Test-Path -LiteralPath $BaselinePath -PathType Leaf)) {
    throw "SemVer baseline configuration not found: $BaselinePath"
}

$baseline = Get-Content -Raw -LiteralPath $BaselinePath | ConvertFrom-Json
if ($baseline.baseline_version -ne $expectedVersion -or $baseline.baseline_tag -ne $expectedTag) {
    throw "SemVer baseline must record version $expectedVersion at tag $expectedTag."
}

$recordedCrates = @($baseline.crates | ForEach-Object { [string]$_ })
if ($recordedCrates.Count -ne 6 -or @($recordedCrates | Sort-Object -Unique).Count -ne 6) {
    throw "SemVer baseline must contain exactly six unique crate names."
}

$metadataJson = & cargo metadata --locked --format-version 1 --no-deps
if ($LASTEXITCODE -ne 0) {
    throw "cargo metadata failed with exit code $LASTEXITCODE."
}
$metadata = $metadataJson | ConvertFrom-Json
$publishableCrates = @(
    $metadata.packages |
        Where-Object { $null -eq $_.publish -or @($_.publish) -contains "crates-io" } |
        ForEach-Object { [string]$_.name }
)

$differences = @(Compare-Object ($recordedCrates | Sort-Object) ($publishableCrates | Sort-Object))
if ($publishableCrates.Count -ne 6 -or $differences.Count -ne 0) {
    $actual = ($publishableCrates | Sort-Object) -join ", "
    throw "Recorded SemVer crates do not match the six publishable workspace crates. Actual: $actual"
}

function Record-InitialBaseline {
    $message = "Initial SemVer API baseline recorded for $($expectedTag): $($recordedCrates -join ', ')"
    Write-Host "::notice title=Initial SemVer API baseline::$message"
    if ($env:GITHUB_STEP_SUMMARY) {
        @"
## Initial Rust SemVer API baseline

$message

Compatibility checks will begin after the v0.1.0 tag exists.
"@ | Add-Content -LiteralPath $env:GITHUB_STEP_SUMMARY
    }
}

Write-Host "Verified SemVer baseline $expectedTag for: $($recordedCrates -join ', ')"
if ($ValidateOnly) {
    return
}

$matchingTags = @(& git tag --list $expectedTag)
if ($LASTEXITCODE -ne 0) {
    throw "Unable to inspect baseline tag $expectedTag (git exit code $LASTEXITCODE)."
}
$tagExists = @($matchingTags | Where-Object { $_ -eq $expectedTag }).Count -eq 1
if (-not $tagExists) {
    if ($RecordInitialBaseline) {
        Record-InitialBaseline
        return
    }
    throw "Cannot run cargo-semver-checks before baseline tag $expectedTag exists."
}
if ($RecordInitialBaseline) {
    throw "Baseline tag $expectedTag already exists; run cargo-semver-checks instead of recording an initial baseline."
}

$releaseTags = @(
    & git tag --merged HEAD --list "v[0-9]*.[0-9]*.[0-9]*" --sort=-version:refname |
        Where-Object { $_ -match '^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$' }
)
if ($LASTEXITCODE -ne 0 -or $releaseTags.Count -eq 0) {
    throw "Unable to resolve the latest stable release tag for SemVer comparison."
}
$comparisonTag = $releaseTags[0]
Write-Host "Comparing the public Rust API with latest release $comparisonTag."
foreach ($crate in $recordedCrates) {
    Write-Host "+ cargo semver-checks --package $crate --baseline-rev $comparisonTag"
    & cargo semver-checks --package $crate --baseline-rev $comparisonTag
    if ($LASTEXITCODE -ne 0) {
        throw "cargo-semver-checks failed for $crate with exit code $LASTEXITCODE."
    }
}
