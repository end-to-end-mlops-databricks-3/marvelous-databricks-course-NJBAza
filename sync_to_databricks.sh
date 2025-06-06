#!/usr/bin/env bash

# Configuration
TARGET_PATH="/Workspace/Users/njavierbuitragoa@gmail.com/marvelous-databricks-course-NJBAza"
TMP_DIR="./.databricks_sync_temp"

echo "Preparing filtered sync folder..."

# Clean previous temp dir if exists
rm -rf "$TMP_DIR"
mkdir "$TMP_DIR"

# Use git to export only tracked (non-ignored) files to the temp folder
git ls-files | tar --create --files-from=- | tar -x -C "$TMP_DIR"

echo "Syncing filtered files to Databricks: $TARGET_PATH"

# Sync the filtered folder
databricks workspace import-dir "$TMP_DIR" "$TARGET_PATH" --overwrite

# Cleanup
rm -rf "$TMP_DIR"

echo "Sync complete."
