"""A2UI template examples for the SQL Explorer agent.

The DataTable custom component handles:
- Dynamic columns (auto-detected from row keys)
- Row rendering with alternating styles
- Pagination controls (Previous/Next buttons)
- Page change actions sent back to the agent executor
"""

SQL_EXPLORER_UI_EXAMPLES = r'''
--- DATATABLE COMPONENT EXAMPLE ---
Use the DataTable custom component for SQL query results.
Replace "__REPLACE_WITH_QUERY_ID__" with the actual query_id from execute_sql_query.

```json
[
  {
    "surfaceUpdate": {
      "surfaceId": "sql-results",
      "components": [
        {
          "id": "root",
          "component": {
            "DataTable": {
              "title": {"path": "/title"},
              "rows": {"path": "/rows"},
              "currentPage": {"path": "/currentPage"},
              "totalPages": {"path": "/totalPages"},
              "totalCount": {"path": "/totalCount"},
              "queryId": {"path": "/queryId"},
              "action": {"name": "page_change"}
            }
          }
        }
      ]
    }
  },
  {
    "dataModelUpdate": {
      "surfaceId": "sql-results",
      "contents": [
        {"key": "title", "valueString": "Query Results"},
        {"key": "queryId", "valueString": "__REPLACE_WITH_QUERY_ID__"},
        {"key": "currentPage", "valueNumber": 1},
        {"key": "totalPages", "valueNumber": 18},
        {"key": "totalCount", "valueNumber": 347},
        {
          "key": "rows",
          "valueArray": [
            {"AlbumId": 1, "Title": "For Those About To Rock", "ArtistId": 1},
            {"AlbumId": 2, "Title": "Balls to the Wall", "ArtistId": 2}
          ]
        }
      ]
    }
  },
  {
    "beginRendering": {
      "surfaceId": "sql-results",
      "root": "root"
    }
  }
]
```
DataTable auto-detects columns from the row keys. No need to specify columns explicitly.

--- PAGINATION DATA UPDATE EXAMPLE ---
When user clicks Previous/Next, send ONLY a dataModelUpdate.
DO NOT resend surfaceUpdate or beginRendering.

The agent executor handles this automatically:
1. Receives userAction with queryId and direction
2. Fetches new page from SQLSessionManager
3. Sends only dataModelUpdate (no UI rebuild)

```json
[
  {
    "dataModelUpdate": {
      "surfaceId": "sql-results",
      "contents": [
        {"key": "currentPage", "valueNumber": 2},
        {
          "key": "rows",
          "valueArray": [
            {"TrackId": 21, "Name": "Next Page Song 1", "Milliseconds": 250000},
            {"TrackId": 22, "Name": "Next Page Song 2", "Milliseconds": 180000}
          ]
        }
      ]
    }
  }
]
```

--- ERROR DISPLAY EXAMPLE ---
Use standard components for error display.

```json
[
  {
    "surfaceUpdate": {
      "surfaceId": "sql-results",
      "components": [
        {
          "id": "root",
          "component": {
            "Card": {
              "child": "error-content"
            }
          }
        },
        {
          "id": "error-content",
          "component": {
            "Column": {
              "children": {"explicitList": ["error-icon", "error-title", "error-message"]},
              "alignment": "center"
            }
          }
        },
        {
          "id": "error-icon",
          "component": {
            "Icon": {
              "name": {"literalString": "error"}
            }
          }
        },
        {
          "id": "error-title",
          "component": {
            "Text": {
              "text": {"literalString": "Query Error"},
              "usageHint": "h3"
            }
          }
        },
        {
          "id": "error-message",
          "component": {
            "Text": {
              "text": {"path": "/errorMessage"}
            }
          }
        }
      ]
    }
  },
  {
    "dataModelUpdate": {
      "surfaceId": "sql-results",
      "contents": [
        {"key": "errorMessage", "valueString": "SQL syntax error: ..."}
      ]
    }
  },
  {
    "beginRendering": {
      "surfaceId": "sql-results",
      "root": "root"
    }
  }
]
```

--- SCHEMA DISPLAY EXAMPLE ---
For showing database schema, use a List with Card items.

```json
[
  {
    "surfaceUpdate": {
      "surfaceId": "sql-results",
      "components": [
        {
          "id": "root",
          "component": {
            "Column": {
              "children": {"explicitList": ["schema-title", "tables-list"]},
              "alignment": "stretch"
            }
          }
        },
        {
          "id": "schema-title",
          "component": {
            "Text": {
              "text": {"literalString": "Database Schema"},
              "usageHint": "h2"
            }
          }
        },
        {
          "id": "tables-list",
          "component": {
            "Column": {
              "children": {
                "template": {
                  "componentId": "table-item",
                  "dataBinding": "/tables"
                }
              },
              "alignment": "stretch"
            }
          }
        },
        {
          "id": "table-item",
          "component": {
            "Card": {
              "child": "table-info"
            }
          }
        },
        {
          "id": "table-info",
          "component": {
            "Column": {
              "children": {"explicitList": ["table-name", "table-details"]},
              "alignment": "start"
            }
          }
        },
        {
          "id": "table-name",
          "component": {
            "Text": {
              "text": {"path": "/name"},
              "usageHint": "h4"
            }
          }
        },
        {
          "id": "table-details",
          "component": {
            "Text": {
              "text": {"path": "/details"},
              "usageHint": "caption"
            }
          }
        }
      ]
    }
  },
  {
    "dataModelUpdate": {
      "surfaceId": "sql-results",
      "contents": [
        {
          "key": "tables",
          "valueArray": [
            {"name": "Album", "details": "AlbumId, Title, ArtistId (347 rows)"},
            {"name": "Artist", "details": "ArtistId, Name (275 rows)"},
            {"name": "Track", "details": "TrackId, Name, AlbumId, ... (3503 rows)"}
          ]
        }
      ]
    }
  },
  {
    "beginRendering": {
      "surfaceId": "sql-results",
      "root": "root"
    }
  }
]
```

'''
