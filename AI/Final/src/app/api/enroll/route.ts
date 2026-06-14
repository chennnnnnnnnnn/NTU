// POST /api/enroll
// Body: { code: string, consent_version: string }
// Behaviour:
//   - Validate code format.
//   - Find existing research_participants row by external_code OR create one.
//   - Stamp consent_version + consented_at if missing.
//   - Write a consent_signed research_events row (idempotent ok, multiple consents allowed).
//   - Return { participant_id }.

import { NextResponse } from "next/server";
import { researchServiceClient } from "@/lib/supabase/research";

const CODE_RE = /^[A-Z0-9]{6,12}$/;

export async function POST(req: Request) {
  if (!process.env.SUPABASE_SERVICE_ROLE_KEY) {
    return errorResponse(503, "SUPABASE_SERVICE_ROLE_KEY is not set on the server.");
  }

  let body: { code?: unknown; consent_version?: unknown };
  try {
    body = await req.json();
  } catch {
    return errorResponse(400, "Body must be JSON.");
  }

  const code = typeof body.code === "string" ? body.code.toUpperCase().trim() : "";
  const consentVersion =
    typeof body.consent_version === "string" && body.consent_version.length > 0
      ? body.consent_version
      : "v1.0";

  if (!CODE_RE.test(code)) {
    return errorResponse(400, "Invalid code format. Must be 6–12 uppercase letters or digits.");
  }

  const supa = researchServiceClient();

  // Upsert behaviour: find existing or insert. (Avoids breaking pre-seeded codes.)
  const { data: existing, error: selErr } = await supa
    .from("research_participants")
    .select("id, consent_version, consented_at, status, fish_voice_id, elevenlabs_voice_id")
    .eq("external_code", code)
    .maybeSingle();

  if (selErr) {
    return errorResponse(500, `Database error: ${selErr.message}`);
  }

  let participantId: string;
  let status: string = "enrolled";
  let fishVoiceId: string | null = null;
  let elevenVoiceId: string | null = null;

  if (existing?.id) {
    participantId = existing.id as string;
    status = (existing.status as string) ?? "enrolled";
    fishVoiceId = (existing.fish_voice_id as string | null) ?? null;
    elevenVoiceId = (existing.elevenlabs_voice_id as string | null) ?? null;
    // Update consent stamp if not yet set or if version changed.
    if (!existing.consented_at || existing.consent_version !== consentVersion) {
      const { error: updErr } = await supa
        .from("research_participants")
        .update({
          consent_version: consentVersion,
          consented_at: new Date().toISOString(),
        })
        .eq("id", participantId);
      if (updErr) {
        return errorResponse(500, `Could not update consent: ${updErr.message}`);
      }
    }
  } else {
    const { data: inserted, error: insErr } = await supa
      .from("research_participants")
      .insert({
        external_code: code,
        consent_version: consentVersion,
        consented_at: new Date().toISOString(),
        status: "enrolled",
      })
      .select("id")
      .single();
    if (insErr || !inserted) {
      return errorResponse(500, `Could not create participant: ${insErr?.message ?? "unknown"}`);
    }
    participantId = inserted.id as string;
  }

  // Audit event (failure here is non-fatal).
  await supa.from("research_events").insert({
    participant_id: participantId,
    event_type: "consent_signed",
    event_payload: { consent_version: consentVersion, code_prefix: code.slice(0, 2), resume_status: status },
    client_ts: new Date().toISOString(),
  });

  // Resume routing — pick the right stage to enter based on participant state.
  const nextPath = routeForStatus({ status, participantId, fishVoiceId, elevenVoiceId });

  return NextResponse.json({ participant_id: participantId, next: nextPath, status });
}

/**
 * Decide which screen the participant should land on next.
 * Mirrors the schema's status enum:
 *   enrolled     → /calibration  (or /processing if voice ids already exist)
 *   pre_done     → /train        (pre-test recordings complete)
 *   train_done   → /post-test
 *   post_done    → /done          (final survey)
 *   delayed_done → /done/complete (fully done)
 *   withdrawn    → /             (start over with a fresh code)
 */
function routeForStatus({
  status,
  participantId,
  fishVoiceId,
  elevenVoiceId,
}: {
  status: string;
  participantId: string;
  fishVoiceId: string | null;
  elevenVoiceId: string | null;
}): string {
  switch (status) {
    case "pre_done":
      return `/train?id=${participantId}`;
    case "train_done":
      return `/post-test?id=${participantId}`;
    case "post_done":
      return `/done?id=${participantId}`;
    case "delayed_done":
      return `/done/complete?id=${participantId}`;
    case "withdrawn":
      return `/`;
    case "enrolled":
    default:
      // If calibration uploaded voice ids but status was never advanced,
      // skip both calibration and processing — the participant is ready
      // for pre-test. /processing is shown only when the user is still on
      // it (it polls and routes forward when ready).
      if (fishVoiceId && elevenVoiceId) {
        return `/pre-test?id=${participantId}`;
      }
      return `/calibration?id=${participantId}`;
  }
}

function errorResponse(status: number, message: string) {
  return NextResponse.json({ error: message }, { status });
}
