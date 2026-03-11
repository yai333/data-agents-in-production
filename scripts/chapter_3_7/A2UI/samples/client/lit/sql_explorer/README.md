# SQL Explorer Client

A Lit-based A2UI client with a custom **DataTable** component for displaying SQL query results with dynamic pagination.

## Custom DataTable Component

The `DataTable` component (`ui/custom-components/data-table.ts`) provides:

- **Dynamic columns** - Auto-detected from row data using `Object.keys(firstRow)`
- **Styled table** - Clean, Material-inspired design
- **Pagination controls** - Previous/Next buttons with page info
- **Server-side search** - Search box that filters results without LLM
- **Action emission** - Dispatches `page_change`, `search`, `clear_search` actions

### Usage in A2UI

```json
{
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
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `title` | BoundValue<string> | Table title |
| `columns` | BoundValue<array> | Optional column definitions |
| `rows` | BoundValue<array> | Row data (array of objects) |
| `currentPage` | BoundValue<number> | Current page number |
| `totalPages` | BoundValue<number> | Total number of pages |
| `totalCount` | BoundValue<number> | Total row count |
| `queryId` | BoundValue<string> | Query ID for pagination |
| `action` | Action | Action config for page changes |

## Running

### Prerequisites

- Node.js 18+
- The SQL Explorer agent running on `http://localhost:10003`

### Start the Client

```bash
cd samples/client/lit/sql_explorer
npm install
npm run dev
```

Opens at `http://localhost:5173` (or next available port)

## Architecture

```
sql_explorer/
├── index.html              # Entry point
├── sql-explorer.ts         # Main app component
├── client.ts               # A2A/A2UI client
├── ui/custom-components/
│   ├── data-table.ts       # DataTable component
│   └── register-components.ts
├── package.json
└── tsconfig.json
```

## How Dynamic Pagination Works

1. **Initial Query**: Client sends text query → Agent generates SQL → Returns DataTable with page 1
2. **Page Change**: User clicks Next → Client sends `userAction` with `page_change`
3. **Instant Response**: Agent fetches next page (no LLM) → Returns `dataModelUpdate` only
4. **UI Update**: DataTable reactively updates with new rows

The key is that pagination **doesn't require LLM** - the agent stores the SQL query and re-executes with different OFFSET.
