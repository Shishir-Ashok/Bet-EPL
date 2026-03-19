"use client";
import { type SortState } from "@/lib/useSortable";

interface SortThProps {
  label: string;
  column: string;
  sort: SortState;
  toggle: (col: string) => void;
  className?: string;
}

export function SortTh({
  label,
  column,
  sort,
  toggle,
  className = "",
}: SortThProps) {
  const active = sort.column === column;
  const isDesc = active && sort.dir === "desc";
  const isAsc = active && sort.dir === "asc";

  return (
    <th
      onClick={() => toggle(column)}
      className={`px-5 py-3 text-left text-xs font-semibold
                  select-none cursor-pointer transition-colors duration-100
                  whitespace-nowrap group
                  ${active ? "text-primary" : "text-muted hover:text-primary"}
                  ${className}`}
    >
      <span className="inline-flex items-center gap-1.5">
        {label}
        <span className="inline-flex flex-col gap-[2px]">
          {/* Up arrow */}
          <svg
            width="7"
            height="4"
            viewBox="0 0 7 4"
            fill="none"
            style={{ opacity: isAsc ? 1 : active ? 0.25 : 0.35 }}
            className="transition-opacity duration-100"
          >
            <path
              d="M3.5 0L7 4H0L3.5 0Z"
              fill={isAsc ? "#3B82F6" : "currentColor"}
            />
          </svg>
          {/* Down arrow */}
          <svg
            width="7"
            height="4"
            viewBox="0 0 7 4"
            fill="none"
            style={{ opacity: isDesc ? 1 : active ? 0.25 : 0.35 }}
            className="transition-opacity duration-100"
          >
            <path
              d="M3.5 4L0 0H7L3.5 4Z"
              fill={isDesc ? "#3B82F6" : "currentColor"}
            />
          </svg>
        </span>
      </span>
    </th>
  );
}
