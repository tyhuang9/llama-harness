[CmdletBinding()]
param(
    [string]$Version
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$expectedRustVersion = "1.88"
$expectedCrates = @(
    "llama-harness",
    "llama-harness-core",
    "llama-harness-evals",
    "llama-harness-observability",
    "llama-harness-ollama",
    "llama-harness-tauri"
) | Sort-Object
$stableVersionPattern = '^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$'
$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path

if ($Version -and $Version -notmatch $stableVersionPattern) {
    throw "Release version '$Version' must be an exact stable SemVer value such as 0.1.0."
}

Push-Location -LiteralPath $repositoryRoot
try {
    $metadataJson = & cargo metadata --locked --format-version 1 --no-deps
    if ($LASTEXITCODE -ne 0) {
        throw "cargo metadata failed with exit code $LASTEXITCODE."
    }
    $metadata = $metadataJson | ConvertFrom-Json
    $publishable = @(
        $metadata.packages |
            Where-Object { @($_.publish) -contains "crates-io" }
    )
    $actualCrates = @($publishable | ForEach-Object { [string]$_.name } | Sort-Object)
    $differences = @(Compare-Object $expectedCrates $actualCrates)
    if ($publishable.Count -ne 6 -or $differences.Count -ne 0) {
        throw "Expected exactly the six supported crates.io packages. Actual: $($actualCrates -join ', ')"
    }

    $metadataVersions = @($publishable | ForEach-Object { [string]$_.version } | Sort-Object -Unique)
    if ($metadataVersions.Count -ne 1 -or $metadataVersions[0] -notmatch $stableVersionPattern) {
        throw "The six crates.io packages must share one stable SemVer version. Actual: $($metadataVersions -join ', ')"
    }
    if (-not $Version) {
        $Version = $metadataVersions[0]
        Write-Host "No requested version supplied; validating Cargo metadata version $Version."
    }
    if ($metadataVersions[0] -ne $Version) {
        throw "Requested version $Version does not match Cargo metadata version $($metadataVersions[0])."
    }

    $invalidMsrv = @(
        $publishable |
            Where-Object { [string]$_.rust_version -ne $expectedRustVersion } |
            ForEach-Object { "$($_.name)=$($_.rust_version)" }
    )
    if ($invalidMsrv.Count -ne 0) {
        throw "Every published crate must declare Rust $expectedRustVersion. Invalid: $($invalidMsrv -join ', ')"
    }

    $changelogPath = Join-Path $repositoryRoot "CHANGELOG.md"
    $changelog = Get-Content -Raw -LiteralPath $changelogPath
    $escapedVersion = [Regex]::Escape($Version)
    $versionHeadings = [Regex]::Matches($changelog, "(?m)^##\s+$escapedVersion(?:\s|$)")
    if ($versionHeadings.Count -ne 1) {
        throw "CHANGELOG.md must contain exactly one level-two heading for $Version."
    }
    $releaseHeading = [Regex]::Match(
        $changelog,
        "(?m)^##\s+$escapedVersion\s+—\s+(?<state>Unreleased|[0-9]{4}-[0-9]{2}-[0-9]{2})\s*$"
    )
    if (-not $releaseHeading.Success) {
        throw "The $Version changelog heading must end with '— Unreleased' or an ISO date."
    }
    $headingState = $releaseHeading.Groups["state"].Value
    if ($headingState -ne "Unreleased") {
        $parsedDate = [DateTime]::MinValue
        if (-not [DateTime]::TryParseExact(
            $headingState,
            "yyyy-MM-dd",
            [Globalization.CultureInfo]::InvariantCulture,
            [Globalization.DateTimeStyles]::None,
            [ref]$parsedDate
        )) {
            throw "The $Version changelog date '$headingState' is not a valid ISO calendar date."
        }
    }
    $bodyStart = $releaseHeading.Index + $releaseHeading.Length
    $nextHeading = [Regex]::Match($changelog.Substring($bodyStart), "(?m)^##\s+")
    $bodyLength = if ($nextHeading.Success) { $nextHeading.Index } else { $changelog.Length - $bodyStart }
    $releaseBody = $changelog.Substring($bodyStart, $bodyLength)
    $meaningfulBody = @(
        $releaseBody -split "`r?`n" |
            Where-Object {
                $line = $_.Trim()
                $line -and -not $line.StartsWith("<!--") -and -not $line.StartsWith("-->")
            }
    )
    if ($meaningfulBody.Count -eq 0) {
        throw "The $Version changelog section must not be empty."
    }

    $workingTree = @(& git status --porcelain=v1 --untracked-files=all)
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect the Git working tree (exit code $LASTEXITCODE)."
    }
    if ($workingTree.Count -ne 0) {
        $preview = @($workingTree | Select-Object -First 20) -join "`n"
        throw "Rust release validation requires a clean Git working tree:`n$preview"
    }

    $sourceCommit = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $sourceCommit -notmatch '^[0-9a-f]{40}$') {
        throw "Unable to resolve the exact source commit for the release."
    }
    Write-Host "Validated Rust release $Version at $sourceCommit."
    Write-Host "Published crates: $($actualCrates -join ', ')"
    Write-Host "MSRV: Rust $expectedRustVersion"
}
finally {
    Pop-Location
}
