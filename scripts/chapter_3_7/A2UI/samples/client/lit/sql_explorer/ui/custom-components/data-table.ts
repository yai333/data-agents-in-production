/**
 * DataTable - A custom A2UI component for displaying SQL query results.
 *
 * Features:
 * - Dynamic columns based on query results
 * - Pagination with Previous/Next buttons
 * - Search/filter functionality
 * - Displays current page and total info
 * - Emits page_change and search actions
 */

import { Root } from "@a2ui/lit/ui";
import { v0_8 } from "@a2ui/lit";
import { html, css, TemplateResult, CSSResultGroup } from "lit";
import { customElement, property, state } from "lit/decorators.js";
import { map } from "lit/directives/map.js";

const StateEvent = v0_8.Events.StateEvent;
type Action = v0_8.Types.Action;

export interface DataTableColumn {
  key: string;
  label?: string;
  width?: string;
}

@customElement("a2ui-data-table")
export class DataTable extends Root {
  // Column definitions - can be explicit or derived from data
  @property({ type: Array }) accessor columns: DataTableColumn[] = [];

  // Row data - array of objects
  @property({ type: Array }) accessor rows: Record<string, unknown>[] = [];

  // Pagination state
  @property({ type: Number }) accessor currentPage: number = 1;
  @property({ type: Number }) accessor totalPages: number = 1;
  @property({ type: Number }) accessor totalCount: number = 0;
  @property({ type: Number }) accessor pageSize: number = 20;

  // Query ID for pagination actions
  @property({ type: String }) accessor queryId: string = "";

  // Title for the table
  @property({ type: String }) accessor title: string = "Query Results";

  // Search term (from server)
  @property({ type: String }) accessor searchTerm: string = "";

  // Enable search UI
  @property({ type: Boolean }) accessor searchable: boolean = true;

  // Action configuration for pagination
  @property({ type: Object }) accessor action: Action | null = null;

  // Local search input state
  @state() accessor #localSearchTerm: string = "";
  @state() accessor #selectedSearchColumn: string = "_all_"; // "_all_" = client-side filter

  #cachedColumns: DataTableColumn[] = [];
  #boundHandleDataUpdate = this.#handleDataUpdate.bind(this);

  override connectedCallback(): void {
    super.connectedCallback();
    window.addEventListener("a2ui-data-update", this.#boundHandleDataUpdate);
  }

  override disconnectedCallback(): void {
    super.disconnectedCallback();
    window.removeEventListener("a2ui-data-update", this.#boundHandleDataUpdate);
  }

  #handleDataUpdate(event: Event) {
    const customEvent = event as CustomEvent<{ surfaceId: string }>;
    if (customEvent.detail?.surfaceId === this.surfaceId) {
      this.requestUpdate();
    }
  }

  /**
   * Resolves a property value that may be a path reference or a literal value.
   * A2UI passes properties as either:
   * - Path references: { path: "/rows" }
   * - Literal values: { literalString: "value" }, { literalNumber: 42 }, etc.
   * - Direct values (already resolved)
   */
  #resolveValue<T>(value: unknown, defaultValue: T): T {
    if (value === null || value === undefined) {
      return defaultValue;
    }

    // Check if it's a path reference
    if (typeof value === "object" && "path" in (value as object)) {
      const pathRef = value as { path: string };
      if (this.processor && this.surfaceId) {
        const resolved = this.processor.getData(
          this.component,
          pathRef.path,
          this.surfaceId
        );
        return (resolved ?? defaultValue) as T;
      }
      return defaultValue;
    }

    // Check if it's a literal value
    if (typeof value === "object") {
      const obj = value as Record<string, unknown>;
      if ("literalString" in obj) return obj.literalString as T;
      if ("literalNumber" in obj) return obj.literalNumber as T;
      if ("literalBoolean" in obj) return obj.literalBoolean as T;
    }

    // It's already a direct value
    return value as T;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  static override styles: any = [css`
      :host {
        display: block;
        font-family: "Roboto", "Arial", sans-serif;
        width: 100%;
      }

      .container {
        display: flex;
        flex-direction: column;
        gap: 16px;
        padding: 16px;
      }

      .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
      }

      .title {
        font-size: 1.25rem;
        font-weight: 500;
        color: #202124;
      }

      .meta {
        font-size: 0.875rem;
        color: #5f6368;
      }

      .table-wrapper {
        overflow-x: auto;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
      }

      table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.875rem;
      }

      thead {
        background: #f8f9fa;
        border-bottom: 2px solid #e0e0e0;
      }

      th {
        padding: 12px 16px;
        text-align: left;
        font-weight: 500;
        color: #202124;
        white-space: nowrap;
      }

      td {
        padding: 12px 16px;
        border-bottom: 1px solid #e0e0e0;
        color: #3c4043;
        max-width: 300px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      tr:hover {
        background: #f8f9fa;
      }

      tr:last-child td {
        border-bottom: none;
      }

      .pagination {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 16px;
        padding: 8px 0;
      }

      .pagination button {
        display: flex;
        align-items: center;
        gap: 4px;
        padding: 8px 16px;
        border: 1px solid #dadce0;
        border-radius: 4px;
        background: #fff;
        color: #1a73e8;
        font-size: 0.875rem;
        cursor: pointer;
        transition: background 0.2s, box-shadow 0.2s;
      }

      .pagination button:hover:not(:disabled) {
        background: #f8f9fa;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
      }

      .pagination button:disabled {
        color: #9aa0a6;
        cursor: not-allowed;
      }

      .pagination button.primary {
        background: #1a73e8;
        color: #fff;
        border-color: #1a73e8;
      }

      .pagination button.primary:hover:not(:disabled) {
        background: #1557b0;
      }

      .pagination button.primary:disabled {
        background: #9aa0a6;
        border-color: #9aa0a6;
      }

      .page-info {
        font-size: 0.875rem;
        color: #5f6368;
      }

      .empty-state {
        padding: 48px;
        text-align: center;
        color: #5f6368;
      }

      .empty-state .icon {
        font-size: 48px;
        margin-bottom: 16px;
        opacity: 0.5;
      }

      .search-bar {
        display: flex;
        gap: 8px;
        margin-bottom: 8px;
      }

      .search-bar input {
        flex: 1;
        padding: 10px 14px;
        font-size: 0.875rem;
        border: 1px solid #dadce0;
        border-radius: 4px;
        outline: none;
        transition: border-color 0.2s, box-shadow 0.2s;
      }

      .search-bar input:focus {
        border-color: #1a73e8;
        box-shadow: 0 0 0 2px rgba(26, 115, 232, 0.2);
      }

      .search-bar button {
        padding: 10px 20px;
        font-size: 0.875rem;
        font-weight: 500;
        color: #fff;
        background: #1a73e8;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        transition: background 0.2s;
      }

      .search-bar button:hover {
        background: #1557b0;
      }

      .search-bar button.clear {
        background: #f1f3f4;
        color: #5f6368;
      }

      .search-bar button.clear:hover {
        background: #e8eaed;
      }

      .search-bar select {
        padding: 10px 14px;
        font-size: 0.875rem;
        border: 1px solid #dadce0;
        border-radius: 4px;
        background: #fff;
        cursor: pointer;
        min-width: 140px;
      }

      .search-bar select:focus {
        border-color: #1a73e8;
        outline: none;
      }

      .search-hint {
        font-size: 0.75rem;
        color: #9aa0a6;
        margin-top: -4px;
      }

      .active-filter {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        background: #e8f0fe;
        border-radius: 4px;
        font-size: 0.875rem;
        color: #1a73e8;
      }

      .active-filter button {
        padding: 2px 6px;
        font-size: 0.75rem;
        background: transparent;
        border: 1px solid #1a73e8;
        border-radius: 4px;
        color: #1a73e8;
        cursor: pointer;
      }

      .active-filter button:hover {
        background: #d2e3fc;
      }
  `];

  render() {
    const resolvedRows = this.#resolveValue<Record<string, unknown>[]>(this.rows, []);
    const resolvedColumns = this.#resolveValue<DataTableColumn[]>(this.columns, []);
    const resolvedCurrentPage = this.#resolveValue<number>(this.currentPage, 1);
    const resolvedTotalPages = this.#resolveValue<number>(this.totalPages, 1);
    const resolvedTotalCount = this.#resolveValue<number>(this.totalCount, 0);
    const resolvedQueryId = this.#resolveValue<string>(this.queryId, "");
    const resolvedTitle = this.#resolveValue<string>(this.title, "Query Results");
    const resolvedSearchTerm = this.#resolveValue<string>(this.searchTerm, "");

    let cols = resolvedColumns.length > 0
      ? resolvedColumns
      : this.#deriveColumns(resolvedRows);

    if (cols.length > 0) {
      this.#cachedColumns = cols;
    } else if (this.#cachedColumns.length > 0) {
      cols = this.#cachedColumns;
    }

    if (resolvedRows.length === 0 && resolvedTotalCount === 0 && !resolvedSearchTerm) {
      return this.#renderEmpty();
    }

    return html`
      <div class="container">
        ${this.#renderHeader(resolvedTitle, resolvedTotalCount)}
        ${this.searchable ? this.#renderSearchBar(cols) : ""}
        ${resolvedSearchTerm ? this.#renderActiveFilter(resolvedSearchTerm) : ""}
        ${this.#renderTable(cols, resolvedRows)}
        ${this.#renderPagination(resolvedCurrentPage, resolvedTotalPages, resolvedQueryId)}
      </div>
    `;
  }

  #deriveColumns(rows: Record<string, unknown>[]): DataTableColumn[] {
    if (rows.length === 0) return [];
    const firstRow = rows[0];
    if (!firstRow || typeof firstRow !== "object") return [];
    return Object.keys(firstRow).map((key) => ({
      key,
      label: key,
    }));
  }

  #renderHeader(title: string, totalCount: number): TemplateResult {
    return html`
      <div class="header">
        <span class="title">${title}</span>
        <span class="meta">${totalCount.toLocaleString()} total rows</span>
      </div>
    `;
  }

  #renderTable(cols: DataTableColumn[], rows: Record<string, unknown>[]): TemplateResult {
    // Apply client-side filtering if "All columns (local)" is selected
    const filteredRows = this.#getFilteredRows(rows);

    return html`
      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              ${map(cols, (col) => html`<th>${col.label || col.key}</th>`)}
            </tr>
          </thead>
          <tbody>
            ${filteredRows.length === 0
        ? html`<tr><td colspan="${cols.length}" style="text-align: center; color: #9aa0a6;">No matching rows</td></tr>`
        : map(
          filteredRows,
          (row) => html`
                    <tr>
                      ${map(cols, (col) => html`<td>${this.#formatCell(row[col.key])}</td>`)}
                    </tr>
                  `
        )
      }
          </tbody>
        </table>
      </div>
    `;
  }

  #getFilteredRows(rows: Record<string, unknown>[]): Record<string, unknown>[] {
    // Only apply client-side filter when "All columns" is selected
    if (this.#selectedSearchColumn !== "_all_" || !this.#localSearchTerm.trim()) {
      return rows;
    }

    const term = this.#localSearchTerm.toLowerCase();
    return rows.filter((row) =>
      Object.values(row).some((value) =>
        String(value).toLowerCase().includes(term)
      )
    );
  }

  #formatCell(value: unknown): string {
    if (value === null || value === undefined) return "—";
    if (typeof value === "object") return JSON.stringify(value);
    return String(value);
  }

  #renderPagination(currentPage: number, totalPages: number, queryId: string): TemplateResult {
    const isFirstPage = currentPage <= 1;
    const isLastPage = currentPage >= totalPages;

    return html`
      <div class="pagination">
        <button
          ?disabled=${isFirstPage}
          @click=${() => this.#handlePageChange("previous", currentPage, queryId)}
        >
          ← Previous
        </button>
        <span class="page-info">
          Page ${currentPage} of ${totalPages}
        </span>
        <button
          class="primary"
          ?disabled=${isLastPage}
          @click=${() => this.#handlePageChange("next", currentPage, queryId)}
        >
          Next →
        </button>
      </div>
    `;
  }

  #renderSearchBar(cols: DataTableColumn[]): TemplateResult {
    const isClientSide = this.#selectedSearchColumn === "_all_";

    return html`
      <div class="search-bar">
        <select
          .value=${this.#selectedSearchColumn}
          @change=${(e: Event) => {
        this.#selectedSearchColumn = (e.target as HTMLSelectElement).value;
      }}
        >
          <option value="_all_">All columns (local)</option>
          ${map(cols, (col) => html`
            <option value="${col.key}">${col.label || col.key}</option>
          `)}
        </select>
        <input
          type="text"
          placeholder=${isClientSide ? "Filter current page..." : `Search in ${this.#selectedSearchColumn}...`}
          .value=${this.#localSearchTerm}
          @input=${(e: Event) => {
        this.#localSearchTerm = (e.target as HTMLInputElement).value;
        // For client-side, filter immediately
        if (isClientSide) {
          this.requestUpdate();
        }
      }}
          @keydown=${(e: KeyboardEvent) => {
        if (e.key === "Enter" && !isClientSide) {
          this.#handleSearch();
        }
      }}
        />
        ${isClientSide
        ? ""
        : html`<button @click=${this.#handleSearch}>Search DB</button>`
      }
      </div>
      <div class="search-hint">
        ${isClientSide
        ? "Filtering current page only (instant)"
        : "Searches database with SQL (may be slow on unindexed columns)"
      }
      </div>
    `;
  }

  #renderActiveFilter(searchTerm: string): TemplateResult {
    return html`
      <div class="active-filter">
        <span>Filtered by: "${searchTerm}"</span>
        <button @click=${this.#handleClearSearch}>Clear</button>
      </div>
    `;
  }

  #renderEmpty(): TemplateResult {
    return html`
      <div class="empty-state">
        <div class="icon">📋</div>
        <div>No data to display</div>
      </div>
    `;
  }

  #handleSearch() {
    if (!this.#localSearchTerm.trim()) return;

    const resolvedQueryId = this.#resolveValue<string>(this.queryId, "");
    const context: Action["context"] = [
      { key: "searchTerm", value: { literalString: this.#localSearchTerm.trim() } },
      { key: "queryId", value: { literalString: resolvedQueryId } },
      { key: "searchColumn", value: { literalString: this.#selectedSearchColumn } },
    ];

    const searchAction: Action = {
      name: "search",
      context,
    };

    const evt = new StateEvent<"a2ui.action">({
      eventType: "a2ui.action",
      action: searchAction,
      dataContextPath: this.dataContextPath,
      sourceComponentId: this.id,
      sourceComponent: this.component,
    });

    this.dispatchEvent(evt);
  }

  #handleClearSearch() {
    this.#localSearchTerm = "";

    const resolvedQueryId = this.#resolveValue<string>(this.queryId, "");
    const context: Action["context"] = [
      { key: "queryId", value: { literalString: resolvedQueryId } },
    ];

    const clearAction: Action = {
      name: "clear_search",
      context,
    };

    const evt = new StateEvent<"a2ui.action">({
      eventType: "a2ui.action",
      action: clearAction,
      dataContextPath: this.dataContextPath,
      sourceComponentId: this.id,
      sourceComponent: this.component,
    });

    this.dispatchEvent(evt);
  }

  #handlePageChange(direction: "previous" | "next", currentPage: number, queryId: string) {
    // Build action context with pagination info
    const context: Action["context"] = [
      { key: "direction", value: { literalString: direction } },
      { key: "queryId", value: { literalString: queryId } },
      { key: "currentPage", value: { literalNumber: currentPage } },
    ];

    const paginationAction: Action = this.action
      ? { ...this.action, context: [...(this.action.context || []), ...context] }
      : { name: "page_change", context };

    // Dispatch the A2UI action event
    const evt = new StateEvent<"a2ui.action">({
      eventType: "a2ui.action",
      action: paginationAction,
      dataContextPath: this.dataContextPath,
      sourceComponentId: this.id,
      sourceComponent: this.component,
    });

    this.dispatchEvent(evt);
  }
}
