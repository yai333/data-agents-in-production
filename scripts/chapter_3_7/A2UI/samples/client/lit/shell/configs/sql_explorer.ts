import { AppConfig } from "./types.js";

export const config: AppConfig = {
  key: "sql_explorer",
  title: "SQL Explorer",
  background: `radial-gradient(
    at 0% 0%,
    light-dark(rgba(129, 140, 248, 0.2), rgba(99, 102, 241, 0.15)) 0px,
    transparent 50%
  ),
  radial-gradient(
    at 100% 100%,
    light-dark(rgba(165, 180, 252, 0.2), rgba(67, 56, 202, 0.1)) 0px,
    transparent 50%
  ),
  linear-gradient(
    120deg,
    light-dark(#f8fafc, #0f172a) 0%,
    light-dark(#e2e8f0, #1e293b) 100%
  )`,
  placeholder: "Show me all albums by AC/DC",
  loadingText: [
    "Querying the database...",
    "Analyzing schema...",
    "Building SQL query...",
    "Fetching results...",
  ],
  serverUrl: "http://localhost:10003",
};
