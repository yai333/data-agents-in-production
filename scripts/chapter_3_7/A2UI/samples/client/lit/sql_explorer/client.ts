import { Part, SendMessageSuccessResponse, Task } from "@a2a-js/sdk";
import { A2AClient } from "@a2a-js/sdk/client";
import { v0_8 } from "@a2ui/lit";

const A2AUI_MIME_TYPE = "application/json+a2aui";
const AGENT_URL = "http://localhost:10003";

export class A2UIClient {
  #serverUrl: string;
  #client: A2AClient | null = null;

  constructor(serverUrl: string = AGENT_URL) {
    this.#serverUrl = serverUrl;
  }

  async #getClient() {
    if (!this.#client) {
      this.#client = await A2AClient.fromCardUrl(
        `${this.#serverUrl}/.well-known/agent-card.json`,
        {
          fetchImpl: async (url, init) => {
            const headers = new Headers(init?.headers);
            headers.set("X-A2A-Extensions", "https://a2ui.org/a2a-extension/a2ui/v0.8");
            return fetch(url, { ...init, headers });
          }
        }
      );
    }
    return this.#client;
  }

  async send(
    message: v0_8.Types.A2UIClientEventMessage | string
  ): Promise<v0_8.Types.ServerToClientMessage[]> {
    const client = await this.#getClient();

    let parts: Part[] = [];

    if (typeof message === 'string') {
      parts = [{ kind: "text", text: message }];
    } else {
      parts = [{
        kind: "data",
        data: message as unknown as Record<string, unknown>,
        mimeType: A2AUI_MIME_TYPE,
      } as Part];
    }

    const response = await client.sendMessage({
      message: {
        messageId: crypto.randomUUID(),
        role: "user",
        parts: parts,
        kind: "message",
      },
    });

    if ("error" in response) {
      throw new Error(response.error.message);
    }

    const result = (response as SendMessageSuccessResponse).result as Task;
    console.log("DEBUG A2A response:", JSON.stringify(result, null, 2));

    if (result.kind === "task" && result.status.message?.parts) {
      const messages: v0_8.Types.ServerToClientMessage[] = [];
      for (const part of result.status.message.parts) {
        console.log("DEBUG part:", JSON.stringify(part, null, 2));
        if (part.kind === 'data') {
          messages.push(part.data as v0_8.Types.ServerToClientMessage);
        }
      }
      return messages;
    }

    return [];
  }
}
