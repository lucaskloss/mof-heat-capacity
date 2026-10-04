#!/usr/bin/env bash

# Shared shell selectors; source after defining PROJECT_DIR.
MOF=mof5
GUEST=ch4
HOST=""
MOLECULE=""
DEFAULT_MD_TEMPERATURES="175,200,225,250,275,300,325,350,375,400,425"
DEFAULT_REPORT_TEMPERATURES="200,225,250,275,300,325,350,375,400"

validate_campaign_selection() {
    GUEST=${GUEST,,}
    if [[ ! "${MOF}" =~ ^[a-z][a-z0-9_-]*$ ]]; then
        echo "error: --mof must be a lowercase label starting with a letter (letters, digits, _ and -)" >&2
        return 2
    fi
    case "${GUEST}" in
        ch4|co2|h2o) ;;
        *) echo "error: --guest must be ch4, co2, or h2o" >&2; return 2 ;;
    esac
    MOF_PATH="${MOF}/"
    # Preserve existing methane/MOF-5 campaign paths and restart locations.
    if [[ "${MOF}/${GUEST}" == mof5/ch4 ]]; then
        MOF_PATH=""
    fi
    if [[ -n "${HOST}" && "${HOST}" != /* ]]; then
        HOST="${PROJECT_DIR}/${HOST}"
    fi
    if [[ -n "${MOLECULE}" && "${MOLECULE}" != /* ]]; then
        MOLECULE="${PROJECT_DIR}/${MOLECULE}"
    fi
}

campaign_host_path() {
    if [[ -n "${HOST}" ]]; then
        printf '%s\n' "${HOST}"
        return
    fi
    local suffix
    for suffix in pdb cif gro extxyz; do
        if [[ -f "${PROJECT_DIR}/input/${MOF}.${suffix}" ]]; then
            printf '%s\n' "${PROJECT_DIR}/input/${MOF}.${suffix}"
            return
        fi
    done
    echo "error: no input/${MOF} structure found; supply --host PATH" >&2
    return 2
}
