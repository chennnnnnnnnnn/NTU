// POST /api/submit-survey
// Body: {
//   participant_id: uuid,
//   likert: { self_efficacy, perceived_improvement, naturalness_c1,
//             naturalness_c2, naturalness_c3, engagement },  values 1..5
//   ranking: [1|2|3, 1|2|3, 1|2|3]   permutation of [1,2,3]
//   comment: string | null
// }
// Survey responses live in research_events with event_type='survey_submitted'
// and the payload above. Also sets status='delayed_done' (final stage marker).

import { NextResponse } from "next/server";
import { researchServiceClient } from "@/lib/supabase/research";

const ID_RE = /^[0-9a-f-]{36}$/i;
const LIKERT_KEYS = [
  "self_efficacy",
  "perceived_improvement",
  "naturalness_c1",
  "naturalness_c2",
  "naturalness_c3",
  "engagement",
] as const;

export async function POST(req: Request) {
  if (!process.env.SUPABASE_SERVICE_ROLE_KEY) {
    return errorResponse(503, "SUPABASE_SERVICE_ROLE_KEY is not set on the server.");
  }

  let body: {
    participant_id?: unknown;
    likert?: unknown;
    ranking?: unknown;
    comment?: unknown;
  };
  try {
    body = await req.json();
  } catch {
    return errorResponse(400, "Body must be JSON.");
  }

  const participantId =
    typeof body.participant_id === "string" ? body.participant_id.trim() : "";
  if (!ID_RE.test(participantId)) return errorResponse(400, "Invalid participant_id.");

  const likertRaw = body.likert;
  if (likertRaw == null || typeof likertRaw !== "object") {
    return errorResponse(400, "likert must be an object.");
  }
  const likert: Record<string, number> = {};
  for (const k of LIKERT_KEYS) {
    const v = (likertRaw as Record<string, unknown>)[k];
    if (typeof v !== "number" || !Number.isInteger(v) || v < 1 || v > 5) {
      return errorResponse(400, `likert.${k} must be an integer 1..5.`);
    }
    likert[k] = v;
  }

  const ranking = body.ranking;
  if (!Array.isArray(ranking) || ranking.length !== 3) {
    return errorResponse(400, "ranking must be an array of length 3.");
  }
  const rankSet = new Set<number>();
  for (const r of ranking) {
    if (![1, 2, 3].includes(r as number)) {
      return errorResponse(400, "ranking must contain only 1, 2, 3.");
    }
    rankSet.add(r as number);
  }
  if (rankSet.size !== 3) {
    return errorResponse(400, "ranking must be a permutation of 1, 2, 3.");
  }

  let comment: string | null = null;
  if (typeof body.comment === "string") {
    comment = body.comment.trim().slice(0, 1000) || null;
  }

  const supa = researchServiceClient();

  // Verify participant exists.
  const { data: participant, error: pErr } = await supa
    .from("research_participants")
    .select("id, status")
    .eq("id", participantId)
    .maybeSingle();
  if (pErr) return errorResponse(500, `Database error: ${pErr.message}`);
  if (!participant) return errorResponse(404, "Participant not found.");

  // Write the survey as a research_events row.
  const { error: eErr } = await supa.from("research_events").insert({
    participant_id: participantId,
    event_type: "survey_submitted",
    event_payload: { likert, ranking, comment },
    client_ts: new Date().toISOString(),
  });
  if (eErr) return errorResponse(500, `Could not save survey: ${eErr.message}`);

  // Update status to delayed_done (final marker; using existing enum value
  // since schema has no separate "completed" state).
  await supa
    .from("research_participants")
    .update({ status: "delayed_done" })
    .eq("id", participantId);

  return NextResponse.json({ ok: true });
}

function errorResponse(status: number, message: string) {
  return NextResponse.json({ error: message }, { status });
}
