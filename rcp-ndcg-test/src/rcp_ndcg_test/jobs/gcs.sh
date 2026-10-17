#!/usr/bin/env bash
# The node scripts' GCS transfer dispatch (one home; wave0.sh and bootstrap.sh source it).
#
# The stock engine image ships neither gcloud nor gsutil. The mounted auth script runs FIRST (it is
# executed, never printed, and its output is never echoed; it sets up the credentials every path uses),
# then the transfer path is chosen once and recorded for the report:
#
#   gcs_sdk_on_path         -> after the auth script: an installed Cloud SDK's bin dir first on PATH,
#                              searched in RCP_GCLOUD_SDK_DIRS (empty: no search)
#   gcs_transfer_detect     -> prints gcloud | gsutil | python
#   gcs_cp SRC DST [dir|file|auto] -> copy a file or directory, either side gs://; for a gs:// source
#                              the kind is required (auto is for local sources)
#   gcs_ls URI              -> list a gs:// prefix
#   gcs_rm URI              -> delete one gs:// object
#
# The Python path installs gcsfs into a tools directory OUTSIDE the engine environment (pip --target,
# the same way uv gets installed), from the wheelhouse when one is staged there and the stage is already
# local, else from PyPI (declared, recorded), and runs gcs.py with it. The tools directory and the
# interpreter come from the caller (GCS_TOOLS_DIR, GCS_PYTHON), so no default hides a machine path.

set -euo pipefail

#: Put a Cloud SDK's bin directory first on PATH, once the auth script ran (it may install the SDK; a
#: child process cannot change our PATH): the first directory of RCP_GCLOUD_SDK_DIRS (colon-separated)
#: that holds gcloud or gsutil. Unset, the list is the SDK's usual install locations below; set EMPTY,
#: nothing is searched (the tests' hermetic PATH: a machine's SDK never stands in for an absent CLI).
gcs_sdk_on_path() {
  local default="$HOME/google-cloud-sdk/bin:/root/google-cloud-sdk/bin:/opt/google-cloud-sdk/bin"
  default+=":/usr/lib/google-cloud-sdk/bin:/usr/local/google-cloud-sdk/bin"
  local dirs="${RCP_GCLOUD_SDK_DIRS-$default}" sdk_bin
  local -a candidates=()
  if [[ -n "$dirs" ]]; then
    IFS=: read -r -a candidates <<<"$dirs"
  fi
  for sdk_bin in "${candidates[@]}"; do
    if [[ -n "$sdk_bin" && (-x "$sdk_bin/gcloud" || -x "$sdk_bin/gsutil") ]]; then
      export PATH="$sdk_bin:$PATH"
      return 0
    fi
  done
  return 0
}

#: The transfer path wave 0 records: gcloud, gsutil, or python (the gcsfs helper).
gcs_transfer_detect() {
  if command -v gcloud >/dev/null; then
    printf 'gcloud\n'
  elif command -v gsutil >/dev/null; then
    printf 'gsutil\n'
  else
    printf 'python\n'
  fi
}

#: Install gcsfs into GCS_TOOLS_DIR (never the engine environment): from a wheelhouse when one is
#: already on local disk (GCS_WHEELHOUSE, the staged one), else from PyPI - the one declared network
#: dependency of the transfer setup, before the stage itself is reachable.
gcs_ensure_tools() {
  local tools="${GCS_TOOLS_DIR:?set GCS_TOOLS_DIR to a tools directory outside the engine environment}"
  if [[ -f "$tools/gcsfs/__init__.py" ]]; then
    return 0
  fi
  mkdir -p "$tools"
  if [[ -n "${GCS_WHEELHOUSE:-}" && -d "$GCS_WHEELHOUSE" ]] && ls "$GCS_WHEELHOUSE"/gcsfs-*.whl >/dev/null 2>&1; then
    "$GCS_PY" -m pip install --quiet --no-index --find-links "$GCS_WHEELHOUSE" --target "$tools" gcsfs
  else
    "$GCS_PY" -m pip install --quiet --target "$tools" gcsfs
  fi
}

#: The python-path invocation of the helper: the tools dir first on sys.path, the engine env untouched.
gcs_run() { # gcs_run GCS_PY_PATH ARGS...
  local helper="${GCS_HELPER_PY:?set GCS_HELPER_PY to the mounted gcs.py}"
  PYTHONPATH="${GCS_TOOLS_DIR}${PYTHONPATH:+:$PYTHONPATH}" "$GCS_PY" "$helper" "$@"
}

gcs_cp() { # gcs_cp SRC DST [dir|file|auto]: copy a file or directory, either side gs://
  # A directory source copies its CONTENTS under DST (the python helper's semantics, so the layout is
  # the same whichever path runs); a file source copies as the destination. One retry: a transient GCS
  # error (connection reset, 5xx) must not cost a wave its logs or report.
  local attempt
  for attempt in 1 2; do
    if _gcs_cp_once "$@"; then
      return 0
    fi
    if ((attempt < 2)); then
      echo "gcs: the copy of $1 failed once; retrying" >&2
      sleep 5
    fi
  done
  return 1
}

_gcs_cp_once() { # the transfer itself, once
  local src="${1%/}" dst="$2"
  # The caller declares the source's kind: "dir" (a local directory or a gs:// prefix; its CONTENTS
  # copy under DST), "file", or "auto" (a local source by -d; anything else is a file). Every path
  # means the same layout for the same kind.
  local kind="${3:-auto}"
  if [[ "$kind" == auto && -d "$src" ]]; then
    kind="dir"
  fi
  case "$(gcs_transfer_detect)" in
    gcloud | gsutil)
      if [[ "$kind" == dir ]]; then
        # The CLIs nest a directory source under an existing destination; gcs.py copies contents. The
        # wildcard form copies contents everywhere: for a local destination it must exist first (gcloud
        # refuses a missing one), for gs:// gcloud creates it.
        [[ "$dst" == gs://* ]] || mkdir -p "$dst"
        if [[ "$src" == gs://* ]]; then
          # A remote prefix's wildcard is expanded by the service, not by this shell.
          if command -v gcloud >/dev/null; then
            gcloud storage cp -r "${src%/}"/* "$dst"
          else
            gsutil -m cp -r "${src%/}"/* "$dst"
          fi
        else
          local had_dotglob had_nullglob
          had_dotglob=$(shopt -p dotglob || true)
          had_nullglob=$(shopt -p nullglob || true)
          shopt -s dotglob nullglob  # dotfiles copy like gcs.py's rglob; an empty dir copies nothing
          local files=("$src"/*)
          [[ -n "$had_dotglob" ]] && eval "$had_dotglob"
          [[ -n "$had_nullglob" ]] && eval "$had_nullglob"
          if ((${#files[@]} == 0)); then
            return 0  # an empty local directory: nothing to copy (the python branch behaves the same)
          fi
          if command -v gcloud >/dev/null; then
            gcloud storage cp -r "${files[@]}" "$dst"
          else
            gsutil -m cp -r "${files[@]}" "$dst"
          fi
        fi
      else
        if command -v gcloud >/dev/null; then
          gcloud storage cp "$src" "$dst"
        else
          gsutil -m cp "$src" "$dst"
        fi
      fi
      ;;
    *) gcs_ensure_tools; gcs_run cp "$src" "$dst" "$kind" ;;
  esac
}

gcs_ls() { # gcs_ls URI: the entries under a gs:// prefix
  case "$(gcs_transfer_detect)" in
    gcloud) gcloud storage ls "$1" ;;
    gsutil) gsutil ls "$1" ;;
    *) gcs_run ls "$1" ;;
  esac
}

gcs_rm() { # gcs_rm URI: delete one gs:// object
  case "$(gcs_transfer_detect)" in
    gcloud) gcloud storage rm "$1" ;;
    gsutil) gsutil rm "$1" ;;
    *) gcs_run rm "$1" ;;
  esac
}
