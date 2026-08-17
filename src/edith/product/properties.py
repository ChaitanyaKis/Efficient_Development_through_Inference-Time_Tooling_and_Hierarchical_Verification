"""A shared vocabulary of product properties, so contradictions are computable.

M4.8 asks for deterministic contradiction detection: a PRD saying "must work offline"
against an architecture requiring a cloud-only service is a contradiction, and the system
should find it without asking a model.

That is impossible over prose. "Must work offline", "no internet required", and "functions
without connectivity" are the same claim in three wordings, and no rule engine reliably
relates them to "depends on a hosted API". Asking an LLM instead reintroduces exactly the
judgement M4.8 says not to rely on.

So the claim is made *structural*. A PRD declares properties from this closed vocabulary; an
architecture declares the properties it actually provides and requires. A contradiction is
then a set intersection against :data:`CONTRADICTORY_PAIRS` -- deterministic, fast, and
impossible to argue with.

The vocabulary is deliberately small. It covers the properties that recur across products
and that genuinely conflict; it is not an ontology of everything a product might be. Prose
still carries the nuance, and a lexical pass supplements this for claims nobody tagged --
but the structural check is the one that can be trusted.
"""

from __future__ import annotations

from enum import StrEnum


class ProductProperty(StrEnum):
    """A structural claim about how a product must or does behave.

    Used by both requirements and architecture so the two can be compared directly.
    """

    # -- Connectivity ---------------------------------------------------------------
    #: Core functionality works with no network.
    OFFLINE_CAPABLE = "OFFLINE_CAPABLE"
    #: Core functionality is unavailable without a network.
    REQUIRES_NETWORK = "REQUIRES_NETWORK"
    #: Depends on a service that only exists as someone else's hosted offering.
    CLOUD_DEPENDENT = "CLOUD_DEPENDENT"
    #: Runs entirely on the user's machine.
    LOCAL_ONLY = "LOCAL_ONLY"

    # -- Identity -------------------------------------------------------------------
    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"
    NO_AUTHENTICATION = "NO_AUTHENTICATION"
    #: Distinguishes what a user may do, not merely who they are.
    AUTHORIZATION_REQUIRED = "AUTHORIZATION_REQUIRED"

    # -- Audience -------------------------------------------------------------------
    MULTI_USER = "MULTI_USER"
    SINGLE_USER = "SINGLE_USER"
    #: Separate tenants whose data must not mix.
    MULTI_TENANT = "MULTI_TENANT"

    # -- Presentation ---------------------------------------------------------------
    MOBILE_RESPONSIVE = "MOBILE_RESPONSIVE"
    DESKTOP_ONLY = "DESKTOP_ONLY"
    #: Meets the stated accessibility standard rather than merely intending to.
    ACCESSIBLE = "ACCESSIBLE"
    #: No user interface at all: a library, a daemon, a CLI.
    HEADLESS = "HEADLESS"

    # -- Data -----------------------------------------------------------------------
    PERSISTENT_STORAGE = "PERSISTENT_STORAGE"
    EPHEMERAL_STORAGE = "EPHEMERAL_STORAGE"
    #: Handles data whose loss or exposure is materially harmful.
    SENSITIVE_DATA = "SENSITIVE_DATA"
    #: Data must remain within a stated jurisdiction.
    DATA_RESIDENCY = "DATA_RESIDENCY"

    # -- Operations -----------------------------------------------------------------
    #: Updates must reach connected users without a refresh.
    REAL_TIME = "REAL_TIME"
    #: Work may complete after the request that started it returns.
    ASYNCHRONOUS = "ASYNCHRONOUS"
    #: Must survive the loss of a single machine.
    HIGH_AVAILABILITY = "HIGH_AVAILABILITY"
    #: Cost, hardware, or footprint is a hard constraint.
    RESOURCE_CONSTRAINED = "RESOURCE_CONSTRAINED"


#: Pairs that cannot both hold. Stored unordered; :func:`conflicts_with` checks both ways.
#:
#: Every entry is a genuine logical conflict, not a stylistic disagreement. "Multi-user" and
#: "single-user" cannot both be true; "asynchronous" and "real-time" can, and are absent.
CONTRADICTORY_PAIRS: frozenset[frozenset[ProductProperty]] = frozenset(
    {
        frozenset({ProductProperty.OFFLINE_CAPABLE, ProductProperty.REQUIRES_NETWORK}),
        frozenset({ProductProperty.OFFLINE_CAPABLE, ProductProperty.CLOUD_DEPENDENT}),
        frozenset({ProductProperty.LOCAL_ONLY, ProductProperty.CLOUD_DEPENDENT}),
        frozenset({ProductProperty.LOCAL_ONLY, ProductProperty.MULTI_TENANT}),
        frozenset(
            {ProductProperty.AUTHENTICATION_REQUIRED, ProductProperty.NO_AUTHENTICATION}
        ),
        # Deciding what a user may do requires knowing who they are.
        frozenset(
            {ProductProperty.AUTHORIZATION_REQUIRED, ProductProperty.NO_AUTHENTICATION}
        ),
        frozenset({ProductProperty.MULTI_USER, ProductProperty.SINGLE_USER}),
        frozenset({ProductProperty.MULTI_TENANT, ProductProperty.SINGLE_USER}),
        frozenset({ProductProperty.MOBILE_RESPONSIVE, ProductProperty.DESKTOP_ONLY}),
        # A product with no interface cannot satisfy an interface requirement.
        frozenset({ProductProperty.HEADLESS, ProductProperty.MOBILE_RESPONSIVE}),
        frozenset({ProductProperty.HEADLESS, ProductProperty.ACCESSIBLE}),
        frozenset({ProductProperty.PERSISTENT_STORAGE, ProductProperty.EPHEMERAL_STORAGE}),
        # Data that must not leave a jurisdiction cannot live in an unconstrained service.
        frozenset({ProductProperty.DATA_RESIDENCY, ProductProperty.CLOUD_DEPENDENT}),
    }
)

#: Properties that *imply* another. Applied transitively before conflict detection, so an
#: architecture that declares MULTI_TENANT is understood to be MULTI_USER without anyone
#: having to remember to tag both.
IMPLICATIONS: dict[ProductProperty, frozenset[ProductProperty]] = {
    ProductProperty.MULTI_TENANT: frozenset(
        {ProductProperty.MULTI_USER, ProductProperty.AUTHENTICATION_REQUIRED}
    ),
    ProductProperty.AUTHORIZATION_REQUIRED: frozenset(
        {ProductProperty.AUTHENTICATION_REQUIRED}
    ),
    ProductProperty.CLOUD_DEPENDENT: frozenset({ProductProperty.REQUIRES_NETWORK}),
    ProductProperty.REAL_TIME: frozenset({ProductProperty.REQUIRES_NETWORK}),
    ProductProperty.DATA_RESIDENCY: frozenset({ProductProperty.SENSITIVE_DATA}),
}

#: Phrases that suggest a property, for the advisory lexical pass over untagged prose. This
#: never produces a blocking finding on its own -- it produces a *hint* that a document
#: should have been tagged, which a human or a reviewing agent can act on.
PROPERTY_HINTS: dict[ProductProperty, tuple[str, ...]] = {
    ProductProperty.OFFLINE_CAPABLE: (
        "offline", "no internet", "without connectivity", "air-gapped", "airgapped",
    ),
    ProductProperty.REQUIRES_NETWORK: ("requires internet", "online only", "network required"),
    ProductProperty.CLOUD_DEPENDENT: (
        "cloud-only", "cloud only", "hosted service", "saas", "managed service",
    ),
    ProductProperty.LOCAL_ONLY: ("local-only", "local only", "on-device", "on device"),
    ProductProperty.AUTHENTICATION_REQUIRED: (
        "authentication", "sign in", "log in", "login", "credentials",
    ),
    ProductProperty.NO_AUTHENTICATION: (
        "no authentication", "no login", "anonymous access", "without signing in",
    ),
    ProductProperty.MULTI_USER: ("multiple users", "multi-user", "collaborators", "team"),
    ProductProperty.SINGLE_USER: ("single user", "single-user", "one user"),
    ProductProperty.MOBILE_RESPONSIVE: ("responsive", "mobile", "small screen", "tablet"),
    ProductProperty.DESKTOP_ONLY: ("desktop-only", "desktop only", "fixed layout"),
    ProductProperty.ACCESSIBLE: ("accessible", "wcag", "screen reader", "a11y"),
    ProductProperty.PERSISTENT_STORAGE: ("persist", "saved", "database", "durable"),
    ProductProperty.SENSITIVE_DATA: (
        "personal data", "pii", "medical", "financial", "confidential",
    ),
    ProductProperty.REAL_TIME: ("real-time", "real time", "live updates", "instantly"),
    ProductProperty.HIGH_AVAILABILITY: ("high availability", "always available", "uptime"),
}


def expand(properties: frozenset[ProductProperty]) -> frozenset[ProductProperty]:
    """Close a property set under :data:`IMPLICATIONS`.

    Applied before conflict detection so an implied property participates. Iterates to a
    fixed point because implications chain: ``MULTI_TENANT`` implies
    ``AUTHENTICATION_REQUIRED``, which nothing further implies, but a future entry might.
    """
    result = set(properties)
    while True:
        additions = {
            implied
            for present in result
            for implied in IMPLICATIONS.get(present, frozenset())
            if implied not in result
        }
        if not additions:
            return frozenset(result)
        result |= additions


def conflicts_with(left: ProductProperty, right: ProductProperty) -> bool:
    """Whether two properties cannot both hold."""
    return frozenset({left, right}) in CONTRADICTORY_PAIRS


def find_conflicts(
    left: frozenset[ProductProperty], right: frozenset[ProductProperty]
) -> tuple[tuple[ProductProperty, ProductProperty], ...]:
    """Every conflicting pair between two property sets, implications included.

    Returns pairs in a stable sorted order so a report does not reshuffle between runs.
    """
    expanded_left = expand(left)
    expanded_right = expand(right)
    found = {
        (first, second)
        for first in expanded_left
        for second in expanded_right
        if conflicts_with(first, second)
    }
    return tuple(sorted(found, key=lambda pair: (pair[0].value, pair[1].value)))


def hints_in(text: str) -> frozenset[ProductProperty]:
    """Properties suggested by free text.

    Advisory only. A hint means "this document talks about offline behaviour but declared no
    property for it", which is worth telling a human. It is never treated as a declaration,
    because a sentence saying a product must *not* work offline contains the word "offline"
    just as clearly as one saying it must.
    """
    lowered = text.lower()
    return frozenset(
        prop
        for prop, phrases in PROPERTY_HINTS.items()
        if any(phrase in lowered for phrase in phrases)
    )
