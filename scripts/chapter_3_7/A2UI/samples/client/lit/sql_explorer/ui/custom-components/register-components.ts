import { componentRegistry } from "@a2ui/lit/ui";
import { DataTable } from "./data-table.js";

export function registerSqlExplorerComponents() {
  // Register DataTable component
  componentRegistry.register("DataTable", DataTable, "a2ui-data-table");

  console.log("Registered SQL Explorer Custom Components: DataTable");
}

export { DataTable };
