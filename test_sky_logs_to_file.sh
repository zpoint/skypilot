#!/bin/bash
# Test script to verify if sky logs completes or terminates early

OUTPUT_FILE="/tmp/sky_logs_test_output.txt"
echo "Starting sky logs test at $(date)"
echo "Writing to: $OUTPUT_FILE"

# Run sky logs and redirect to file
timeout 120 sky logs t-long-setup-run-by-56 1 > "$OUTPUT_FILE" 2>&1
EXIT_CODE=$?

echo ""
echo "==============================================="
echo "Test completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo "==============================================="
echo ""

# Count lines
LINE_COUNT=$(wc -l < "$OUTPUT_FILE")
echo "Total lines received: $LINE_COUNT"

# Show file size
FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
echo "File size: $FILE_SIZE"

# Show first 20 lines
echo ""
echo "First 20 lines:"
head -20 "$OUTPUT_FILE"

# Show last 20 lines
echo ""
echo "Last 20 lines:"
tail -20 "$OUTPUT_FILE"

# Check if it ends properly
echo ""
if tail -5 "$OUTPUT_FILE" | grep -q "Job finished"; then
    echo "✓ Log completed successfully (found 'Job finished' message)"
else
    echo "✗ Log did NOT complete (no 'Job finished' message found)"
fi

echo ""
echo "Expected ~501,605 lines based on server logs"
echo "Actual lines received: $LINE_COUNT"
echo "Missing lines: $((501605 - LINE_COUNT))"
