# Changelog

## Unreleased

- Add CPU/Gloo regression tests requiring coordinated policy shutdown and exactly-once
  command handling at the final-sync deadline.
- Coordinate the final command-wait decision across policy ranks to prevent an
  unmatched shutdown collective while preserving final command handling.
