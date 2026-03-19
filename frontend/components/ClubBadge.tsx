"use client";

interface ClubBadgeProps {
  crest?: string | null;
  tla: string;
  name?: string;
  size?: "sm" | "md" | "lg";
  showName?: boolean;
  className?: string;
}

const sizes = {
  sm: { img: "w-4 h-4", text: "w-5 h-5 text-[8px]" },
  md: { img: "w-6 h-6", text: "w-6 h-6 text-[9px]" },
  lg: { img: "w-8 h-8", text: "w-8 h-8 text-[10px]" },
};

export function ClubBadge({
  crest,
  tla,
  name,
  size = "sm",
  showName = false,
  className = "",
}: ClubBadgeProps) {
  const sz = sizes[size];

  return (
    <span className={`inline-flex items-center gap-1.5 ${className}`}>
      {crest ? (
        <img
          src={crest}
          alt={tla}
          className={`${sz.img} object-contain flex-shrink-0`}
          onError={(e) => {
            // If crest fails to load, swap to TLA fallback
            const el = e.currentTarget;
            el.style.display = "none";
            el.nextElementSibling?.removeAttribute("style");
          }}
        />
      ) : null}
      {/* TLA fallback — hidden when crest loads, shown if crest fails or missing */}
      <span
        className={`${sz.text} rounded font-bold font-mono
                    bg-subtle border border-border text-muted
                    flex items-center justify-center flex-shrink-0`}
        style={{ display: crest ? "none" : "flex" }}
      >
        {tla}
      </span>
      {showName && (
        <span className="font-medium text-primary">{name || tla}</span>
      )}
    </span>
  );
}
