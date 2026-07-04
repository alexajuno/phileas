# Session 01 — auth token strategy

**User:** I'm wiring up login in `src/auth/session.py`. Leaning toward JWTs so the API stays stateless. Thoughts?

**Assistant:** JWTs are stateless, but the cost is revocation: once issued, a JWT is valid until it expires. Logout, a leaked token, or a breach response all need you to invalidate immediately, and you can't without a denylist, which puts state back in and defeats the point.

**User:** Right, and we definitely need instant logout and a kill-switch for compromised sessions. So statelessness isn't worth it here.

**Assistant:** Then opaque tokens backed by a sessions table fit better. The token is a random id, the row holds the state, and deleting the row revokes instantly.

**User:** Let's go with opaque server-side tokens, not JWTs. Record this decision so we don't relitigate it in three months, the revocation requirement is the whole reason.

**Assistant:** Will do. Implementing the sessions table and the lookup middleware now.

**User:** Also bump the lint timeout in CI, it keeps flaking. But that's unrelated.
