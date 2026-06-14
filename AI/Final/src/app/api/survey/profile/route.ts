// POST /api/survey/profile
// Body: { participant_id, age_bracket?, gender?, native_language?,
//         english_level?, used_headphones? }
// Behaviour:
//   - Save demographics on survey_participants.
//   - Materialise the random clip assignment (survey_responses) if not present.
//   - Advance status to in_progress.
//   - Return { ok, next }.

import { NextResponse } from "next/server";
import { surveyServiceClient } from "@/lib/supabase/survey";
import { getOrCreateSurveyItems } from "@/lib/survey-items";
import { PROFILE_FIELDS } from "@/lib/survey-config";

const ID_RE = /^[0-9a-f-]{36}$/i;

export async function POST(req: Request) {
  if (!process.env.SUPABASE_SERVICE_ROLE_KEY) {
    return errorResponse(503, "SUPABASE_SERVICE_ROLE_KEY is not set on the server.");
  }

  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return errorResponse(400, "Body must be JSON.");
  }

  const participantId = typeof body.participant_id === "string" ? body.participant_id.trim() : "";
  if (!ID_RE.test(participantId)) {
    return errorResponse(400, "Invalid participant_id.");
  }

  // Whitelist + validate demographic values against survey-config options.
  const update: Record<string, string | boolean | null> = {};
  for (const field of PROFILE_FIELDS) {
    const raw = body[field.key];
    if (raw == null || raw === "") continue;
    const value = String(raw);
    const allowed = field.options.some((o) => o.value === value);
    if (!allowed) {
      return errorResponse(400, `Invalid value for ${field.key}: ${value}`);
    }
    update[field.key] = value;
  }
  if (typeof body.used_headphones === "boolean") {
    update.used_headphones = body.used_headphones;
  }

  const supa = surveyServiceClient();

  // Confirm participant exists.
  const { data: participant, error: pErr } = await supa
    .from("survey_participants")
    .select("id, status")
    .eq("id", participantId)
    .maybeSingle();
  if (pErr) return errorResponse(500, `Database error: ${pErr.message}`);
  if (!participant) return errorResponse(404, "Participant not found.");

  update.status = "in_progress";
  const { error: uErr } = await supa
    .from("survey_participants")
    .update(update)
    .eq("id", participantId);
  if (uErr) return errorResponse(500, `Could not save profile: ${uErr.message}`);

  // Build the randomised item sequence (idempotent).
  try {
    await getOrCreateSurveyItems(participantId);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return errorResponse(503, msg);
  }

  return NextResponse.json({ ok: true, next: `/clip-survey/rate?id=${participantId}` });
}

function errorResponse(status: number, message: string) {
  return NextResponse.json({ error: message }, { status });
}
