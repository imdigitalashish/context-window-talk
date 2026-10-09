// Candidate "parts" of the knowledge map: a Stack Exchange site (optionally a tag) each; `extra` lists
// further tags whose top-voted questions are merged in. build-map.mjs picks the final selection;
// fetch-corpus.mjs fetches all candidates.
export const CANDIDATE_PARTS = [
  { id: 'react', label: 'React & frontend', site: 'stackoverflow', tagged: 'reactjs' },
  { id: 'css', label: 'CSS & styling', site: 'stackoverflow', tagged: 'css' },
  { id: 'auth', label: 'Auth & login', site: 'stackoverflow', tagged: 'authentication' },
  { id: 'postgres', label: 'Postgres & SQL', site: 'stackoverflow', tagged: 'postgresql' },
  { id: 'docker', label: 'Docker & deploy', site: 'stackoverflow', tagged: 'docker' },
  // The stripe-payments tag also carries unrelated build errors (Kotlin, cURL), so keep payment titles only.
  { id: 'payments', label: 'Stripe & payments', site: 'stackoverflow', tagged: 'stripe-payments',
    keep: /stripe|pay|checkout|subscri|card|invoice|billing|charge|refund|webhook|customer|price|pricing|coupon|plan\b|payout|transfer|connect|intent|3d secure|sca\b|elements|currenc|tax|receipt/i },
  { id: 'ml', label: 'Machine learning', site: 'stackoverflow', tagged: 'machine-learning' },
  { id: 'android', label: 'Android', site: 'stackoverflow', tagged: 'android' },
  { id: 'testing', label: 'Unit testing', site: 'stackoverflow', tagged: 'unit-testing' },
  { id: 'ux', label: 'UX & design', site: 'ux' },
  { id: 'security', label: 'Security', site: 'security' },
  { id: 'law', label: 'Law', site: 'law' },
  { id: 'money', label: 'Personal finance', site: 'money' },
  { id: 'interpersonal', label: 'Relationships', site: 'interpersonal' },
  { id: 'workplace', label: 'Workplace', site: 'workplace' },
  { id: 'cooking', label: 'Cooking', site: 'cooking' },
  { id: 'writing', label: 'Writing & copy', site: 'writing' },
];
