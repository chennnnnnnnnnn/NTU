// POST /api/survey/answer
// Body: { participant_id, item_id, rating }   (rating 1..5)
// Behaviour:
//   - Record the 1–5 answer on the survey_items row (comparison or attention).
//   - When every item is answered → grade the attention checks
//     (attention_passed = all attention items where rating === expected),
//     mark the participant completed.
//   - Return { ok, completed, next }.

import { NextResponse } from "next/server";
import { surveyServiceClient } from "@/lib/supabase/survey";

const ID_RE = /^[0-9a-f-]{36}$/i;

export async function POST(req: Request) {
  if (!process.env.SUPABASE_SERVICE_ROLE_KEY) {
    return errorResponse(503, "SUPABASE_SERVICE_ROLE_KEY is not set on the server.");
  }

  let body: { participant_id?: unknown; item_id?: unknown; rating?: unknown };
  try {
    body = await req.json();
  } catch {
    return errorResponse(400, "Body must be JSON.");
  }

  const participantId = typeof body.participant_id === "string" ? body.participant_id.trim() : "";
  const itemId = typeof body.item_id === "string" ? body.item_id.trim() : "";
  const rating = Number(body.rating);
  if (!ID_RE.test(participantId)) return errorResponse(400, "Invalid participant_id.");
  if (!ID_RE.test(itemId)) return errorResponse(400, "Invalid item_id.");
  if (!Number.isInteger(rating) || rating < 1 || rating > 5) {
    return errorResponse(400, "rating must be an integer 1..5.");
  }

  const supa = surveyServiceClient();

  const { data: updated, error: uErr } = await supa
    .from("survey_items")
    .update({ rating, answered_at: new Date().toISOString() })
    .eq("id", itemId)
    .eq("participant_id", participantId)
    .select("id")
    .maybeSingle();
  if (uErr) return errorResponse(500, `Database error: ${uErr.message}`);
  if (!updated) return errorResponse(404, "Item not found for this participant.");

  // Remaining unanswered?
  const { data: remaining, error: rErr } = await supa
    .from("survey_items")
    .select("id")
    .eq("participant_id", participantId)
    .is("answered_at", null);
  if (rErr) return errorResponse(500, `Database error: ${rErr.message}`);

  const completed = (remaining?.length ?? 0) === 0;
  if (completed) {
    // Grade attention checks.
    const { data: checks } = await supa
      .from("survey_items")
      .select("rating, expected")
      .eq("participant_id", participantId)
      .eq("item_type", "attention");
    const attentionPassed = (checks ?? []).every((c) => c.rating === c.expected);

    await supa
      .from("survey_participants")
      .update({ status: "completed", attention_passed: attentionPassed })
      .eq("id", participantId);
  }

  return NextResponse.json({
    ok: true,
    completed,
    next: completed ? `/clip-survey/done?id=${participantId}` : null,
  });
}

function errorResponse(status: number, message: string) {
  return NextResponse.json({ error: message }, { status });
}
