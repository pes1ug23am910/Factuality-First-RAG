# JarvisLabs execution contract

This guide defines the checked deployment contract for JarvisLabs Linux GPU
instances. It does **not** claim that a remote experiment has been launched or
completed. `requirements/jarvislabs-cp310.txt` is a pinned candidate constraint
set and preflight contract, not a hash-locked or remotely validated Linux
environment.
It intentionally inherits Torch and CUDA from the selected JarvisLabs PyTorch
template instead of installing the Windows-specific Torch wheel used by the
repository's separate Windows lock.

## Prerequisites

- Install and authenticate `jl`; verify with `jl status --json`.
- Install `rsync` on the local machine. JarvisLabs directory targets require it.
- Use a Linux GPU instance whose PyTorch template provides CPython 3.10 and a
  CUDA-enabled Torch build.
- Install JDK 21 below the canonical runtime root described next and expose it
  through `JAVA_HOME` in the instance/startup environment. Do not put
  credentials in startup commands, checked-in files, or run logs.

### Prepare a clean upload bundle

Do not run a directory target from the development worktree. The JarvisLabs
directory sync does not use `.gitignore` as an upload policy, so ignored local
files can otherwise be copied to the instance. In particular, never upload
`.env`, hidden local workspace directories, private process records, or a
project virtual environment.

Build the upload directory from a clean committed tree with `git archive`.
This both excludes ignored/untracked private files and makes the remote package
bytes reproducible even though the directory sync excludes `.git/`. On the
Windows machine that starts and monitors the managed runs:

```powershell
$jarvisStatus = git status --porcelain=v1 --untracked-files=all
if ($LASTEXITCODE -ne 0) { throw "git status failed." }
if ($jarvisStatus) { throw "Commit or otherwise resolve every worktree change before export." }

$jarvisForbidden = git -c core.quotePath=false ls-tree -r --name-only HEAD | Where-Object {
    $isPrivateFile = $_ -match (
        '(^|/)(\.env($|[./])|\.envrc$|[^/]+\.(pem|key|p12|pfx)$|' +
        '[^/]*(PROMPT|SESSION|STATUS|TRACKER|TRANSCRIPT|INSTRUCTION|' +
        'PROGRESS|MEMORY)[^/]*\.(md|txt)$)'
    )
    $isHiddenDirectory = $_ -match '(^|/)\.(?!github(?:/|$))[^/]+/'
    $isPrivateFile -or $isHiddenDirectory
}
if ($LASTEXITCODE -ne 0) { throw "git ls-tree failed." }
if ($jarvisForbidden) { throw "HEAD contains a path forbidden from Jarvis uploads." }

# This path denylist does not replace a secret-content scan of the committed tree.
# Review scanner findings without printing matched credential values.

$jarvisCommit = git rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw "git rev-parse failed." }
$jarvisBundle = Join-Path (
    [System.IO.Path]::GetTempPath()
) ("factuality-rag-" + $jarvisCommit.Substring(0, 12))
$jarvisArchive = "${jarvisBundle}.tar"
if (
    (Test-Path -LiteralPath $jarvisBundle) -or
    (Test-Path -LiteralPath $jarvisArchive)
) {
    throw "Refusing to reuse an existing Jarvis bundle or archive."
}
$null = New-Item -ItemType Directory -Path $jarvisBundle -ErrorAction Stop
git -c core.autocrlf=false archive --format=tar --output=$jarvisArchive HEAD
if ($LASTEXITCODE -ne 0) { throw "git archive failed." }
tar -xf $jarvisArchive -C $jarvisBundle
if ($LASTEXITCODE -ne 0) { throw "archive extraction failed." }
$jarvisUvSource = Join-Path $jarvisBundle "requirements/jarvislabs-uv.toml"
$jarvisUvActive = Join-Path $jarvisBundle "uv.toml"
if (-not (Test-Path -LiteralPath $jarvisUvSource -PathType Leaf)) {
    throw "The committed Jarvis uv configuration is missing from the export."
}
if (Test-Path -LiteralPath $jarvisUvActive) {
    throw "Refusing to replace an unexpected root uv.toml in the export."
}
Copy-Item -LiteralPath $jarvisUvSource -Destination $jarvisUvActive -ErrorAction Stop
Set-Location -LiteralPath $jarvisBundle -ErrorAction Stop
```

The preceding block runs in PowerShell only. From this point through artifact
download and lifecycle cleanup, run every `bash` fence and every `jl` command
from one authenticated POSIX shell. On Windows, use the WSL installation of
`jl`, translate the exported bundle path to `/mnt/<drive>/...`, and `cd` there
before any `jl run .` command. Do not paste the single-quoted POSIX commands
below into PowerShell.

All local `jl run .` commands below must be issued from that exported directory,
not from the original checkout. Confirm that the export contains
`pyproject.toml`, `uv.toml`, `factuality_rag/`, `scripts/`, and
`requirements/jarvislabs-cp310.txt` before starting a billable run. The active
`uv.toml` is deliberately copied only in this clean bundle. Do not add it to
the development checkout: its Jarvis-only exclusions would make an ordinary
local uv project environment omit the required Torch packages.
Because `git archive` exports `HEAD` only, all readiness changes must be
committed first; untracked files are silently absent from the bundle. In
particular, verify that `.gitattributes`, both Jarvis requirements files, both
Jarvis scripts, `factuality_rag/reproducibility.py`, and packaged resources are
present in the export.

## Select or create a running instance

Inspect account state, existing instances, and current availability with
read-only commands first:

```bash
jl status --json
jl list --json
jl gpus --json
```

Reuse a suitable running instance if one exists. Creating or resuming a machine
starts billing, so stop and obtain explicit operator approval immediately before
either action. Only after approval, a new container can be created separately:

```bash
jl create --gpu L4 --region <region> --template pytorch --storage <storage-gb> \
  --name <instance-name> --yes --json
```

Do not use a one-shot `jl run --gpu` for this workflow: the persistent JDK and
environment file must be provisioned before preflight. Record the returned
`machine_id`, wait until that machine is running, and then continue below.

## Provision the persistent runtime

JarvisLabs defines the persistent remote home as `/home/` for containers and
`/home/<user>/` for VMs. Derive one runtime root from that authoritative
`$HOME`; do not hard-code either layout in project commands. Create the root and
a non-secret environment file once on the running instance:

```bash
jl exec <machine_id> -- sh -lc 'set -eu; root="$HOME/factuality-rag-runtime"; mkdir -p "$root/cache/huggingface/datasets" "$root/config" "$root/data" "$root/indexes" "$root/runs" "$root/uploads"; umask 077; printf "%s\n" "export FACTUALITY_RAG_RUNTIME_ROOT=\"$root\"" "export JAVA_HOME=\"$root/jdk-21\"" "export HF_HOME=\"$root/cache/huggingface\"" "export HF_DATASETS_CACHE=\"$root/cache/huggingface/datasets\"" "export PATH=\"\$JAVA_HOME/bin:\$PATH\"" > "$root/config/jarvis-env.sh"; printf "FACTUALITY_RAG_RUNTIME_ROOT=%s\n" "$root"'
```

Reserve `$HOME/factuality-rag-runtime/jdk-21` for JDK 21. Upload a verified
Linux x86-64 archive, verify the same SHA-256 on the instance, and install it
without overwriting an existing runtime. Confirm first that the archive has one
top-level directory; `--strip-components=1` relies on that topology:

```bash
jl upload <machine_id> <absolute-local-jdk21-linux-x64.tar.gz> <absolute-runtime-root>/uploads/jdk-21.tar.gz --json
jl exec <machine_id> -- sh -lc 'set -eu; root="$HOME/factuality-rag-runtime"; archive="$root/uploads/jdk-21.tar.gz"; printf "%s  %s\n" "<expected-sha256>" "$archive" | sha256sum -c -; test ! -e "$root/jdk-21"; top_count="$(tar -tzf "$archive" | awk -F/ "NF {print \$1}" | sort -u | wc -l)"; test "$top_count" -eq 1; temp="$(mktemp -d "$root/.jdk-21.XXXXXX")"; trap "rm -rf -- \"$temp\"" 0; tar -xzf "$archive" --strip-components=1 -C "$temp"; test -x "$temp/bin/java"; "$temp/bin/java" -version 2>&1 | grep -Eq "version \"21([.]|\")"; mv "$temp" "$root/jdk-21"'
jl exec <machine_id> -- sh -lc 'set -eu; . "$HOME/factuality-rag-runtime/config/jarvis-env.sh"; nvidia-smi && test -x "$JAVA_HOME/bin/java" && "$JAVA_HOME/bin/java" -version'
```

Every managed run below explicitly sources `jarvis-env.sh` with `--setup`.
Exports performed by an earlier `jl exec` command do not persist into the
managed run shell. Keep credentials out of this environment file and all setup
commands.

The first `jl exec` command prints the concrete absolute runtime root. Record
that value as `<absolute-runtime-root>` before `jl upload`. Shell variables such
as `$HOME` are not expanded inside YAML config values, and `jl upload` and
`jl download` do not interpret remote shell variables in path arguments. Use
literal absolute remote paths for uploads, configs, and downloads, plus an
absolute local download destination outside the upload bundle. Keep corpora,
indexes, configs, run outputs, JDK files, and model caches only under this one
remote root.

## Mandatory preflight

Run from the clean exported directory on the local machine and replace
placeholders with the live machine ID. Always use the directory target (`.`),
and always pass the Jarvis requirements file explicitly:

```bash
jl run . --script scripts/jarvis_preflight.py --on <machine_id> \
  --requirements requirements/jarvislabs-cp310.txt \
  --setup '. "$HOME/factuality-rag-runtime/config/jarvis-env.sh"' --json --yes
```

The command returns JSON immediately. Record its `run_id` and `machine_id`.
JarvisLabs 0.2.17 creates a `--system-site-packages` venv and installs the
requirements with uv. The bundle-root `uv.toml` excludes Torch, Torchvision,
Torchaudio, and Triton from that resolution so the coherent CUDA stack supplied
by the PyTorch template remains visible. Without this config, uv can install a
newer project-local Torch while inheriting the template's Torchvision, producing
an ABI/operator mismatch. Treat a missing `uv.toml` as a deployment error.
The script itself performs no model or dataset download. It fails nonzero unless
the exact bundle-only uv exclusion config is present with the expected content
and Linux, CPython 3.10, the pinned non-Torch Python distributions, CUDA Torch,
bitsandbytes, Accelerate, Transformers, FAISS, Pyserini, Pyjnius, Java 21, and
`JAVA_HOME` are ready. It also rejects a project-venv copy of Torch that shadows
the template build and executes a tiny in-memory bitsandbytes 4-bit CUDA forward
to surface kernel/runtime mismatches before any model download.

For the no-model mock demo only, use the lighter profile. Passing this check does
not certify the real GPU/Java runtime:

```bash
jl run . --script scripts/jarvis_preflight.py --on <machine_id> \
  --requirements requirements/jarvislabs-cp310.txt \
  --setup '. "$HOME/factuality-rag-runtime/config/jarvis-env.sh"' \
  --json --yes -- --mock
```

Do not omit `--requirements`: directory auto-detection would install only the
base project metadata, while real mode requires the `quantization` extra. Do not
use command mode for setup; command mode starts in `~`, does not support
`--requirements`, and would break project-relative paths. The requirements file
contains `-e .[quantization]`; uv resolves that path from its current working
directory, not from the requirements file's parent. The directory-target
contract supplies the repository root. For manual installation, `cd` to the
exported project root, verify that its copied `uv.toml` is present, then
invoke `uv pip install -r requirements/jarvislabs-cp310.txt` from the activated
`--system-site-packages` environment.

## Run and monitor

After the full matching preflight succeeds, run the idempotent sample smoke with
the same managed environment:

```bash
jl run . --script scripts/run_sample_experiment.sh --on <machine_id> \
  --requirements requirements/jarvislabs-cp310.txt \
  --setup '. "$HOME/factuality-rag-runtime/config/jarvis-env.sh"' --json --yes
```

Record its `run_id` and `machine_id` and monitor it with the bounded commands
below. Success requires the real production Lucene/BM25 query to return valid
hits before the mock pipeline step completes. It does not replace the later
one-query real-model experiment.

For real experiments, first place the model cache, corpus, indexes, copied
config, and output directory under `<absolute-runtime-root>`, then put that
literal root in every config path. Model and dataset acquisition is a separate,
networked preparation step; the preflight does not perform it or prove that a
model fits the selected GPU.

For example, replace the placeholder before saving the copied YAML; do not
leave `$HOME` or an environment-variable reference in these values:

```yaml
index:
  faiss_out: "<absolute-runtime-root>/indexes/wiki100k.faiss"
  corpus_path: "<absolute-runtime-root>/data/wiki_100000_chunks.jsonl"
  pyserini_out: "<absolute-runtime-root>/indexes/wiki100k_lucene"
pipeline:
  runs_dir: "<absolute-runtime-root>/runs"
```

The generator and gating probe share the quantized generator model on CUDA. The
NLI scorer defaults to CPU for predictable memory use; when the selected GPU has
enough free VRAM, enable its batched GPU path explicitly in the copied config:

```yaml
scorer:
  device: "cuda:0"
  nli_batch_size: 8
```

Reduce `nli_batch_size` first if NLI inference runs out of memory, or set
`device: "cpu"` to retain the safe default.

NLI inputs use the strictest available limit from the 512-token application
cap, `tokenizer.model_max_length`, and the model's positional limit. Evidence
premises may be truncated, but the query/claim hypothesis is never silently
truncated. If the complete hypothesis, pair special tokens, and required
evidence space cannot fit, scoring fails before model inference. This also
applies to custom NLI models with limits below 512; shorten or deliberately
reformulate the claim rather than relying on implicit truncation.

Model and tokenizer first loads are synchronized independently. Concurrent
pipeline initialization therefore reuses one cached object for identical load
settings instead of racing two large loads; incompatible settings still fail
closed and require an explicit registry clear.

Start with one real query before scaling the sample. The copied config below
must contain absolute remote corpus/index paths and a supported dataset adapter:

```bash
jl run . --script factuality_rag/experiment_runner.py --on <machine_id> \
  --requirements requirements/jarvislabs-cp310.txt \
  --setup '. "$HOME/factuality-rag-runtime/config/jarvis-env.sh"' \
  --json --yes -- \
  --config <absolute-runtime-root>/config/jarvis-exp.yaml \
  --sample 1 --runs-dir <absolute-runtime-root>/runs
```

Increase `--sample` only after this one-query run exits successfully and its
artifacts have been inspected.

For a longer paid run, assign a recognizable run prefix. The experiment logs
its exact durable run directory before loading models:

```bash
jl run . --script factuality_rag/experiment_runner.py --on <machine_id> \
  --requirements requirements/jarvislabs-cp310.txt \
  --setup '. "$HOME/factuality-rag-runtime/config/jarvis-env.sh"' \
  --json --yes -- \
  --config <absolute-runtime-root>/config/jarvis-exp.yaml \
  --sample 500 --runs-dir <absolute-runtime-root>/runs --run-id nq500
```

If that managed job terminates after writing some predictions, launch the same
code, config, dataset selection, sample size, seed, overrides, and runtime mode
with `--resume` pointing to the exact experiment run directory from the log. Do
not also pass `--run-id`:

```bash
jl run . --script factuality_rag/experiment_runner.py --on <machine_id> \
  --requirements requirements/jarvislabs-cp310.txt \
  --setup '. "$HOME/factuality-rag-runtime/config/jarvis-env.sh"' \
  --json --yes -- \
  --config <absolute-runtime-root>/config/jarvis-exp.yaml \
  --sample 500 --resume <absolute-runtime-root>/runs/<experiment-run-directory>
```

Resume manifest v2 uses its self-hash to reject manifest corruption. Separate
binding comparisons reject changed config, query/reference sequence, runtime
settings, selected library versions, Git state, or runtime package contents.
Each invocation hashes the package-relative paths and exact bytes of every
non-bytecode regular file under `factuality_rag/`; Python cache directories and
bytecode are excluded, while symlinks, non-regular entries, unstable snapshots,
and `.env`/`.env.*` files are rejected. It also verifies that
`import factuality_rag` resolves to the package tree being hashed.

Git HEAD and dirty state are hard resume bindings when Git is available, so a
Git-state change blocks resume even when package bytes match. In a clean archive
export where `.git/` is absent, both launch and resume instead bind the same
`git-not-available` state plus the exact runtime-source digest. The selected
library binding currently records FAISS, Datasets, Transformers, and Sentence
Transformers versions; the mandatory preflight separately checks the wider
Python, Torch/CUDA, quantization, and Java stack. A v1 manifest is intentionally
not accepted by the v2 runner. The checkpoint reader discards only an incomplete
final JSONL fragment and never silently accepts gaps, reordered rows, or a
malformed completed line.

Use bounded polling from the same local machine that started the run:

```bash
jl run logs <run_id> --tail 30
jl run status <run_id> --json
jl run logs <run_id> --tail 50
```

Check once after roughly 15 seconds for setup/import failures, then poll every
60–120 seconds for short runs or 180–300 seconds for longer experiments. Never
request the entire log and do not use `--follow` in automated polling. A success
footer with exit code 0 is the completion condition; on failure, inspect the
bounded tail and status before starting a new run.

When finished, download required artifacts from their persistent `$HOME` path
to an absolute local directory outside the upload bundle:

```bash
jl download <machine_id> <absolute-runtime-root>/runs <absolute-local-results-directory-outside-bundle> -r
```

Replace `<absolute-runtime-root>` with the exact value printed during setup;
do not pass a literal `$HOME` to the local `jl download` command. Verify the
downloaded evidence, then stop and obtain explicit operator approval immediately
before pausing. Only after that approval, run:

```bash
jl pause <machine_id> --yes --json
```

Destroy is deliberately not included as cleanup: it irreversibly deletes the
persistent runtime root. A separately requested destroy requires verified
evidence download and explicit operator approval immediately before the action.
