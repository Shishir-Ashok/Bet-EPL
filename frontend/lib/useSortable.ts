"use client";

import { useState, useMemo } from "react";

export type SortDir = "asc" | "desc";

export interface SortState {
  column: string | null;
  dir: SortDir;
}

// Columns that default to ascending on first click (text / alphabetical)
const ASC_FIRST_COLUMNS = new Set([
  "team_name",
  "_home",
  "_away",
  "_season",
  "action",
]);

function defaultDir(column: string): SortDir {
  return ASC_FIRST_COLUMNS.has(column) ? "asc" : "desc";
}

export function useSortable<T>(
  data: T[],
  initialColumn?: string,
  initialDir?: SortDir,
) {
  const [sort, setSort] = useState<SortState>({
    column: initialColumn ?? null,
    dir: initialDir ?? (initialColumn ? defaultDir(initialColumn) : "desc"),
  });

  function toggle(column: string) {
    setSort((prev) => {
      if (prev.column !== column) {
        // First click on a new column — use the best default direction
        return { column, dir: defaultDir(column) };
      }
      // Already on this column — flip direction
      return { column, dir: prev.dir === "desc" ? "asc" : "desc" };
    });
  }

  const sorted = useMemo(() => {
    if (!sort.column) return data;
    return [...data].sort((a: any, b: any) => {
      const av = a[sort.column!];
      const bv = b[sort.column!];
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      const cmp =
        typeof av === "string"
          ? av.localeCompare(bv, undefined, { sensitivity: "base" })
          : Number(av) - Number(bv);
      return sort.dir === "asc" ? cmp : -cmp;
    });
  }, [data, sort.column, sort.dir]);

  return { sorted, sort, toggle };
}
