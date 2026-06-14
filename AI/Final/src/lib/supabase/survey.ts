// survey.ts — Service-role wrapper restricted to `survey_*` tables.
//
// Mirror of research.ts (migration 001 study). All audio-survey backend code
// MUST go through this wrapper instead of calling createServiceClient()
// directly, so a shared-project service role can never accidentally touch
// mirror-app's or the shadow study's tables.

import { createServiceClient } from "./server";

type SurveyTable =
  | "survey_participants"
  | "survey_clips"
  | "survey_responses"
  | "survey_items";

/**
 * Service-role Supabase client locked to the `survey_*` table set via a
 * runtime guard around `.from()`.
 */
export function surveyServiceClient() {
  const raw = createServiceClient();
  return {
    from(table: SurveyTable) {
      if (!table.startsWith("survey_")) {
        throw new Error(
          `[survey.ts] forbidden table: ${table}. Only survey_* tables are allowed.`,
        );
      }
      return raw.from(table);
    },
    storage: raw.storage,
    rpc: raw.rpc.bind(raw),
  };
}
