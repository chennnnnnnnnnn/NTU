// Supabase server-side client.
// Two variants:
//   (1) createClient(): user-scoped anon client with cookies (for Server Components / actions)
//   (2) createServiceClient(): service-role client — UNRESTRICTED, must NEVER reach the browser.
//
// Service-role usage MUST be confined to `research_*` tables in this project.
// See lib/supabase/research.ts wrapper.

import { createServerClient } from '@supabase/ssr'
import { createClient as createSupabaseClient } from '@supabase/supabase-js'
import { cookies } from 'next/headers'

export async function createClient() {
  const cookieStore = await cookies()

  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll()
        },
        setAll(cookiesToSet) {
          try {
            cookiesToSet.forEach(({ name, value, options }) =>
              cookieStore.set(name, value, options),
            )
          } catch {
            // Server Component context — cookie mutations not allowed; safe to ignore.
          }
        },
      },
    },
  )
}

/**
 * Service-role client. Bypasses RLS. Use ONLY inside server-only code
 * (Route Handlers, Server Actions, server-only utilities) for operations that
 * must work without an authenticated user — e.g. seeding research_participants
 * via invitation code, generating signed upload URLs, writing research_events
 * for anonymous participants.
 *
 * Wrapper functions should restrict all queries to `research_*` tables to
 * avoid accidental writes to mirror-app's `public.*` tables.
 */
export function createServiceClient() {
  return createSupabaseClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!,
    {
      auth: { autoRefreshToken: false, persistSession: false },
    },
  )
}
