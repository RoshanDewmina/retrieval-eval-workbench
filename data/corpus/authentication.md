# Authentication and sessions

Meridian Support Service uses passwordless email links for operator sign-in. A sign-in link expires after fifteen minutes and can be used only once.

An operator session expires after eight hours of inactivity. Administrators can revoke an active session from the access console.

Service tokens are for server-to-server integrations only. Store them in a secret manager and rotate a token immediately if it appears in a client log.
