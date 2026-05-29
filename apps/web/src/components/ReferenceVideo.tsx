import { useState } from "react";

type ReferenceVideoProps = {
  signId: string;
};

export default function ReferenceVideo({ signId }: ReferenceVideoProps) {
  const [available, setAvailable] = useState(true);
  if (!available) return null;

  return (
    <video
      src={`/references/${signId}.webm`}
      controls
      playsInline
      preload="metadata"
      onError={() => setAvailable(false)}
      style={{ width: "100%", maxWidth: "360px", marginTop: "0.75rem", background: "#000", borderRadius: "8px" }}
    />
  );
}
