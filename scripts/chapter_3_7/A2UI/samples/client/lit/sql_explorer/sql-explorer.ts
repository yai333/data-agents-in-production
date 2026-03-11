/*
 Copyright 2025 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

/**
 * SQL Explorer Client Application
 *
 * Demonstrates dynamic pagination with A2UI and the custom DataTable component.
 */

import { SignalWatcher } from "@lit-labs/signals";
import { provide } from "@lit/context";
import { LitElement, html, css, nothing, unsafeCSS } from "lit";
import { customElement, state } from "lit/decorators.js";
import { repeat } from "lit/directives/repeat.js";
import { v0_8 } from "@a2ui/lit";
import * as UI from "@a2ui/lit/ui";
import { A2UIClient } from "./client.js";
import { registerSqlExplorerComponents } from "./ui/custom-components/register-components.js";
import { theme as defaultTheme } from "./theme/default-theme.js";

// Register custom components
registerSqlExplorerComponents();

@customElement("sql-explorer-app")
export class SQLExplorerApp extends SignalWatcher(LitElement) {
  @provide({ context: UI.Context.themeContext })
  accessor theme: v0_8.Types.Theme = defaultTheme;

  @state()
  accessor #requesting = false;

  @state()
  accessor #error: string | null = null;

  @state()
  accessor #inputValue = "";

  // Prevent duplicate pagination requests
  #pendingPagination = false;

  static styles = [
    unsafeCSS(v0_8.Styles.structuralStyles),
    css`
      :host {
        display: block;
        max-width: 1200px;
        margin: 0 auto;
        min-height: 100vh;
        font-family: "Roboto", "Arial", sans-serif;
      }

      .app-header {
        padding: 24px;
        background: linear-gradient(135deg, #1a73e8 0%, #4285f4 100%);
        color: white;
        text-align: center;
      }

      .app-header h1 {
        margin: 0 0 8px 0;
        font-size: 1.75rem;
        font-weight: 500;
      }

      .app-header p {
        margin: 0;
        opacity: 0.9;
        font-size: 0.95rem;
      }

      .content {
        padding: 24px;
      }

      .query-form {
        display: flex;
        gap: 12px;
        margin-bottom: 24px;
      }

      .query-form input {
        flex: 1;
        padding: 12px 16px;
        font-size: 1rem;
        border: 1px solid #dadce0;
        border-radius: 8px;
        outline: none;
        transition: border-color 0.2s, box-shadow 0.2s;
      }

      .query-form input:focus {
        border-color: #1a73e8;
        box-shadow: 0 0 0 2px rgba(26, 115, 232, 0.2);
      }

      .query-form button {
        padding: 12px 24px;
        font-size: 1rem;
        font-weight: 500;
        color: white;
        background: #1a73e8;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: background 0.2s;
      }

      .query-form button:hover:not(:disabled) {
        background: #1557b0;
      }

      .query-form button:disabled {
        background: #9aa0a6;
        cursor: not-allowed;
      }

      .loading {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 48px;
        color: #5f6368;
      }

      .loading .spinner {
        width: 24px;
        height: 24px;
        border: 3px solid #e0e0e0;
        border-top-color: #1a73e8;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-right: 12px;
      }

      @keyframes spin {
        to {
          transform: rotate(360deg);
        }
      }

      .error {
        padding: 16px;
        margin-bottom: 16px;
        background: #fce8e6;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        color: #c5221f;
      }

      .surfaces {
        display: flex;
        flex-direction: column;
        gap: 16px;
      }

      .examples {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 24px;
      }

      .example-btn {
        padding: 8px 16px;
        font-size: 0.875rem;
        color: #1a73e8;
        background: #e8f0fe;
        border: none;
        border-radius: 16px;
        cursor: pointer;
        transition: background 0.2s;
      }

      .example-btn:hover {
        background: #d2e3fc;
      }

      .intro {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 24px;
      }

      .intro h2 {
        margin: 0 0 12px 0;
        font-size: 1.1rem;
        font-weight: 500;
        color: #202124;
      }

      .intro p {
        margin: 0;
        color: #5f6368;
        line-height: 1.6;
      }
    `,
  ];

  #processor = v0_8.Data.createSignalA2uiMessageProcessor();
  #a2uiClient = new A2UIClient();

  #exampleQueries = [
    "Show me all tables",
    "List all albums by AC/DC",
    "Find tracks longer than 5 minutes",
    "Top 10 customers by spending",
    "Show all artists",
  ];

  render() {
    return html`
      <div class="app-header">
        <h1>SQL Explorer</h1>
        <p>Query the Chinook database with natural language</p>
      </div>
      <div class="content">
        ${this.#renderIntro()}
        ${this.#renderExamples()}
        ${this.#renderQueryForm()}
        ${this.#renderError()}
        ${this.#renderLoading()}
        ${this.#renderSurfaces()}
      </div>
    `;
  }

  #renderIntro() {
    const surfaces = this.#processor.getSurfaces();
    if (surfaces.size > 0) return nothing;

    return html`
      <div class="intro">
        <h2>Dynamic Datatable Demo</h2>
        <p>
          This demo shows A2UI's dynamic pagination feature. When you query
          the database, results are displayed with pagination. 
        </p>
      </div>
    `;
  }

  #renderExamples() {
    return html`
      <div class="examples">
        ${this.#exampleQueries.map(
      (q) => html`
            <button class="example-btn" @click=${() => this.#setQuery(q)}>
              ${q}
            </button>
          `
    )}
      </div>
    `;
  }

  #renderQueryForm() {
    return html`
      <form class="query-form" @submit=${this.#handleSubmit}>
        <input
          type="text"
          placeholder="Ask a question about the music database..."
          .value=${this.#inputValue}
          @input=${(e: Event) => {
        this.#inputValue = (e.target as HTMLInputElement).value;
      }}
          ?disabled=${this.#requesting}
        />
        <button type="submit" ?disabled=${this.#requesting || !this.#inputValue}>
          ${this.#requesting ? "Querying..." : "Query"}
        </button>
      </form>
    `;
  }

  #renderError() {
    if (!this.#error) return nothing;
    return html`<div class="error">${this.#error}</div>`;
  }

  #renderLoading() {
    if (!this.#requesting) return nothing;
    return html`
      <div class="loading">
        <div class="spinner"></div>
        <span>Generating SQL and fetching results...</span>
      </div>
    `;
  }

  #renderSurfaces() {
    const surfaces = this.#processor.getSurfaces();
    console.log("DEBUG renderSurfaces: size=", surfaces.size, "requesting=", this.#requesting);
    if (surfaces.size === 0) return nothing;

    return html`
      <div class="surfaces">
        ${repeat(
      surfaces,
      ([surfaceId]) => surfaceId,
      ([surfaceId, surface]) => {
        console.log("DEBUG rendering surface:", surfaceId, surface);
        return html`
              <a2ui-surface
                @a2uiaction=${this.#handleAction}
                .surfaceId=${surfaceId}
                .surface=${surface}
                .processor=${this.#processor}
                .enableCustomElements=${true}
              ></a2ui-surface>
            `;
      }
    )}
      </div>
    `;
  }

  #setQuery(query: string) {
    this.#inputValue = query;
  }

  async #handleSubmit(evt: Event) {
    evt.preventDefault();
    if (!this.#inputValue) return;

    await this.#sendAndProcess(this.#inputValue);
  }

  async #handleAction(evt: v0_8.Events.StateEvent<"a2ui.action">) {
    const [target] = evt.composedPath();
    if (!(target instanceof HTMLElement)) return;

    // Resolve action context
    const context: v0_8.Types.A2UIClientEventMessage["userAction"]["context"] = {};
    const surfaceId = (evt.detail.sourceComponent as unknown as { surfaceId?: string })?.surfaceId || "sql-results";

    if (evt.detail.action.context) {
      for (const item of evt.detail.action.context) {
        if (item.value.literalBoolean !== undefined) {
          context[item.key] = item.value.literalBoolean;
        } else if (item.value.literalNumber !== undefined) {
          context[item.key] = item.value.literalNumber;
        } else if (item.value.literalString !== undefined) {
          context[item.key] = item.value.literalString;
        } else if (item.value.path) {
          const path = this.#processor.resolvePath(
            item.value.path,
            evt.detail.dataContextPath
          );
          const value = this.#processor.getData(
            evt.detail.sourceComponent,
            path,
            surfaceId
          );
          context[item.key] = value;
        }
      }
    }

    const message: v0_8.Types.A2UIClientEventMessage = {
      userAction: {
        surfaceId,
        name: evt.detail.action.name,
        sourceComponentId: target.id,
        timestamp: new Date().toISOString(),
        context,
      },
    };

    const dataOnlyActions = ["page_change", "search", "clear_search"];
    if (dataOnlyActions.includes(evt.detail.action.name)) {
      await this.#sendDataUpdateRequest(message);
    } else {
      await this.#sendAndProcess(message);
    }
  }

  async #sendAndProcess(
    message: v0_8.Types.A2UIClientEventMessage | string
  ) {
    try {
      this.#requesting = true;
      this.#error = null;
      this.#processor.clearSurfaces();

      const messages = await this.#a2uiClient.send(message);
      this.#processor.processMessages(messages);
      this.requestUpdate();
    } catch (err) {
      this.#error = String(err);
    } finally {
      this.#requesting = false;
    }
  }

  async #sendDataUpdateRequest(message: v0_8.Types.A2UIClientEventMessage) {
    if (this.#pendingPagination) return;
    this.#pendingPagination = true;

    try {
      this.#error = null;
      const messages = await this.#a2uiClient.send(message);
      this.#processor.processMessages(messages);

      for (const msg of messages) {
        if (msg.dataModelUpdate?.surfaceId) {
          window.dispatchEvent(new CustomEvent("a2ui-data-update", {
            detail: { surfaceId: msg.dataModelUpdate.surfaceId }
          }));
        }
      }
    } catch (err) {
      this.#error = String(err);
    } finally {
      this.#pendingPagination = false;
    }
  }
}
