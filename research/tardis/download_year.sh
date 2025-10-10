#!/bin/bash
# Download one year of Deribit options data, split by month
#
# Usage:
#   ./download_year.sh 2024 BTC call
#
# Arguments:
#   $1: Year (e.g., 2024)
#   $2: Assets (e.g., BTC or BTC,ETH)
#   $3: Option type (call, put, or both) - optional, defaults to 'both'

set -e  # Exit on error

YEAR=${1:-2024}
ASSETS=${2:-BTC}
OPTION_TYPE=${3:-both}

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║             DOWNLOADING FULL YEAR OF DERIBIT OPTIONS DATA                ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Year: $YEAR"
echo "  Assets: $ASSETS"
echo "  Option type: $OPTION_TYPE"
echo "  Days to expiry: 0-3 days"
echo "  Sampling: 5s"
echo ""

# Check if tardis-machine is running
echo "Checking tardis-machine server..."
if ! curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo "❌ ERROR: tardis-machine server not running!"
    echo ""
    echo "Please start it first:"
    echo "  npx tardis-machine --port=8000"
    echo ""
    exit 1
fi
echo "✓ tardis-machine is running"
echo ""

# Array of months
declare -a MONTHS=(
    "01" "02" "03" "04" "05" "06"
    "07" "08" "09" "10" "11" "12"
)

# Month names for display
declare -a MONTH_NAMES=(
    "January" "February" "March" "April" "May" "June"
    "July" "August" "September" "October" "November" "December"
)

# Calculate days in each month (accounting for leap years)
function days_in_month() {
    local year=$1
    local month=$2

    case $month in
        01|03|05|07|08|10|12) echo 31 ;;
        04|06|09|11) echo 30 ;;
        02)
            # Check leap year
            if [ $((year % 4)) -eq 0 ] && { [ $((year % 100)) -ne 0 ] || [ $((year % 400)) -eq 0 ]; }; then
                echo 29
            else
                echo 28
            fi
            ;;
    esac
}

TOTAL_MONTHS=${#MONTHS[@]}
COMPLETED=0
FAILED=0

echo "════════════════════════════════════════════════════════════════════════════"
echo "Starting downloads: $TOTAL_MONTHS months"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

for i in "${!MONTHS[@]}"; do
    MONTH="${MONTHS[$i]}"
    MONTH_NAME="${MONTH_NAMES[$i]}"
    DAYS=$(days_in_month $YEAR $MONTH)

    FROM_DATE="${YEAR}-${MONTH}-01"
    TO_DATE="${YEAR}-${MONTH}-${DAYS}"

    MONTH_NUM=$((i + 1))

    echo "────────────────────────────────────────────────────────────────────────────"
    echo "[$MONTH_NUM/$TOTAL_MONTHS] $MONTH_NAME $YEAR ($FROM_DATE to $TO_DATE)"
    echo "────────────────────────────────────────────────────────────────────────────"

    START_TIME=$(date +%s)

    if uv run python tardis_download.py \
        --from-date "$FROM_DATE" \
        --to-date "$TO_DATE" \
        --assets "$ASSETS" \
        --option-type "$OPTION_TYPE" \
        --max-days 3 \
        --resample-interval 5s \
        --output-format parquet; then

        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        COMPLETED=$((COMPLETED + 1))

        echo ""
        echo "✓ $MONTH_NAME completed in ${ELAPSED}s"
        echo ""
    else
        FAILED=$((FAILED + 1))
        echo ""
        echo "✗ $MONTH_NAME FAILED!"
        echo ""
        echo "Do you want to continue with remaining months? (y/n)"
        read -r CONTINUE
        if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
            echo "Aborted by user"
            exit 1
        fi
    fi
done

echo "════════════════════════════════════════════════════════════════════════════"
echo "DOWNLOAD COMPLETE"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Summary:"
echo "  Total months: $TOTAL_MONTHS"
echo "  Completed: $COMPLETED"
echo "  Failed: $FAILED"
echo ""
echo "Output files:"
ls -lh datasets_deribit_options/deribit_options_${YEAR}-*.parquet 2>/dev/null || echo "  No files found"
echo ""
