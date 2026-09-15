# SwapAI documentation platform

**Decision:** use **Fumadocs** in the SwapAI repository and publish a static build at `https://stuartmccamley.com/swapai`.

Fumadocs is the best fit because SwapAI is a TypeScript npm library, its documentation will be mostly code examples, and the existing website already uses Next.js 16, React 19 and Tailwind 4. Current Fumadocs package metadata targets Next.js 16 and React 19.2, while its official CLI supports both existing Next applications and a static-export template. Its code blocks provide Shiki highlighting, titles and copy buttons; tabs can persist and synchronise a reader's choice; Twoslash can show real TypeScript types and errors. It also provides local Orama search, same-site versioning and source-level UI control under an MIT licence. [CLI](https://www.fumadocs.dev/docs/cli), [static build](https://www.fumadocs.dev/docs/deploying/static), [code blocks](https://www.fumadocs.dev/docs/ui/components/codeblock), [tabs](https://www.fumadocs.dev/docs/ui/components/tabs), [Twoslash](https://www.fumadocs.dev/docs/markdown/twoslash), [search](https://www.fumadocs.dev/docs/headless/search/orama), [versioning](https://www.fumadocs.dev/docs/navigation), [customisation](https://www.fumadocs.dev/docs/guides/customize-ui), [licence](https://github.com/fuma-nama/fumadocs/blob/dev/LICENSE), [npm package](https://www.npmjs.com/package/fumadocs-ui)

## Recommended deployment

Keep the docs source with the package, for example under `apps/docs`, rather than coupling documentation changes to the currently dirty `stuartmccamley.com` repository.

Configure the docs build with:

```ts
const nextConfig = {
  basePath: "/swapai",
  output: "export",
};
```

Next.js applies `basePath` to pages and assets at build time, and Fumadocs supports static output plus a browser-side search index. Serve the exported files from a small static container and proxy `/swapai` and `/swapai/*` to it. This matches the existing site's use of path rewrites for sibling services while keeping SwapAI independently buildable and deployable. [Next.js `basePath`](https://nextjs.org/docs/app/api-reference/config/next-config-js/basePath), [Fumadocs static search](https://www.fumadocs.dev/docs/deploying/static)

The alternative is adding Fumadocs directly to the existing Next application. That removes one build, but puts SwapAI's docs in the wrong repository and makes package documentation releases depend on the personal site's release. The separate static build is the cleaner boundary.

## Comparison

| Platform | `/swapai` hosting and current site | Code examples | Search | Design control | Versioning | Ownership and work required |
|---|---|---|---|---|---|---|
| **Fumadocs** | Next.js `basePath`; static export or direct integration with an existing Next app | Shiki, copy, titles, diffs, line marking, persistent tabs, package-install blocks and optional Twoslash | Built-in Orama; supports static browser-side indexes | Several polished layouts; CSS/JSX options; CLI can copy components into the repo | Built-in folder-based version selector or separate deployments | MIT and self-hosted. One separate static build is required for the recommended repository boundary. |
| **Nextra** | Native Next plugin; `contentDirBasePath: "/swapai"`; static export supported | Highlighting, titles, copy and persistent tabs | Pagefind, but requires a post-build indexing command | Good docs theme; custom CSS or a completely custom React theme | No first-party versioning feature found in its official docs; it would need manual routes/navigation | MIT and self-hosted. Easy to embed in the existing site, but it changes the site's global Next config and docs still belong in another repo. |
| **Starlight** | Astro supports `base: "/swapai"` and produces static files in `dist`; it cannot plug directly into the Next app | Excellent Expressive Code blocks with copy, editor/terminal frames, markers and synchronised tabs | Pagefind is enabled without configuration | Strong CSS variables, Tailwind support and component replacement | No built-in versioning documented; community plugins exist | MIT and self-hosted. Lowest static-hosting burden, but introduces Astro alongside the existing Next stack. |
| **Docusaurus** | Explicit `baseUrl: "/swapai/"`; always a separate React static site | Strong highlighted blocks, titles, line marking and tabs | Official Algolia support; local search is community-maintained | CSS plus React component wrapping/ejection; deep changes add upgrade work | Best built-in versioning here | MIT and self-hosted. Mature, but the heaviest setup for a small TypeScript library. |
| **Mintlify** | First-class subpath hosting, but the existing domain must proxy `/swapai/*` to Mintlify | Excellent copyable blocks, diffs, line marking and synchronised code groups | Hosted search included | Attractive defaults and branding controls | Built-in version selector | Normal operation is hosted SaaS. Self-hosting requires Enterprise and a substantial platform deployment, so this adds vendor dependence without solving a hard SwapAI problem. |

Sources: [Nextra content path](https://nextra.site/docs/file-conventions/content-directory), [Nextra code blocks](https://nextra.site/docs/guide/syntax-highlighting), [Nextra tabs](https://nextra.site/docs/built-ins/tabs), [Nextra search](https://nextra.site/docs/guide/search), [Nextra static export](https://nextra.site/docs/guide/static-exports), [Nextra custom theme](https://nextra.site/docs/custom-theme), [Nextra licence](https://github.com/shuding/nextra/blob/main/license); [Astro base path](https://docs.astro.build/en/reference/configuration-reference/#base), [Starlight code](https://starlight.astro.build/guides/authoring-content/#code-blocks), [Starlight tabs](https://starlight.astro.build/components/tabs/), [Starlight search](https://starlight.astro.build/guides/site-search/), [Starlight styling](https://starlight.astro.build/guides/css-and-tailwind/), [Starlight licence](https://github.com/withastro/starlight/blob/main/LICENSE); [Docusaurus base URL and deployment](https://docusaurus.io/docs/deployment), [Docusaurus code blocks](https://docusaurus.io/docs/markdown-features/code-blocks), [Docusaurus search](https://docusaurus.io/docs/search), [Docusaurus versioning](https://docusaurus.io/docs/versioning), [Docusaurus customisation](https://docusaurus.io/docs/swizzling), [Docusaurus licence](https://github.com/facebook/docusaurus/blob/main/LICENSE); [Mintlify subpath proxy](https://www.mintlify.com/docs/deploy/reverse-proxy), [Mintlify code](https://www.mintlify.com/docs/create/code), [Mintlify code groups](https://www.mintlify.com/docs/components/code-groups), [Mintlify versions](https://www.mintlify.com/docs/organize/navigation#versions), [Mintlify self-hosting](https://www.mintlify.com/docs/deploy/self-host)

## Why not Starlight?

Starlight is the runner-up. It is simpler to export, Pagefind works immediately, and Expressive Code is excellent. Choose it if the main goal becomes the smallest possible static-docs service. Fumadocs wins for SwapAI because its TypeScript code tooling, native Next.js fit, built-in version navigation and installable UI components are stronger, while static export keeps the runtime just as simple.

“Beautiful” is subjective. The recommendation is based on the current official Fumadocs interface plus its ability to change or own the components, not on a claim that one default theme is objectively prettier.
