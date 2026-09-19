# Security policy

QuditsOnQubits is research software for quantum compilation and experiments. It is
not intended for production security-critical use.

## Supported versions

Only the latest tagged release receives security fixes on a best-effort basis.
Earlier releases are not supported with security backports.

## Report a vulnerability privately

Open the repository's [Security tab](https://github.com/slysek/QuditsOnQubits/security)
and select **Report a vulnerability** under **Advisories**, if that option is
available. This uses GitHub's private vulnerability reporting channel.

If the option is unavailable, contact the maintainer privately using the contact
listed in the [Code of Conduct's Enforcement section](CODE_OF_CONDUCT.md#enforcement).
Do not post vulnerability details, provider tokens, or credentials in public issues.

Include the affected package version or commit, Python version, a minimal
reproduction, and the potential impact. For artifact-loading or backend-related
issues, describe the input format or provider integration without including secrets.

We aim to acknowledge reports within 14 days. This is a best-effort target for a
small research project, not a guaranteed response time. Investigation and fixes
depend on maintainer availability and the issue's scope; disclosure timing can be
discussed privately with the reporter.
