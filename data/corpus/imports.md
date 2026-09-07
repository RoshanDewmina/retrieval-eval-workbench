# Importing service records

Use a CSV file encoded as UTF-8 with a header row. The required columns are case_id, customer_region, opened_at, and summary.

The import preview rejects rows with a missing case_id or an opened_at value that is not ISO 8601. Rejected rows remain available for download for seven days.

Submitting the same import key within twenty-four hours returns the original import result. Corrected data must use a new import key.
