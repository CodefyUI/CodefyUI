import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'CodefyUI',
  tagline: 'A visual, node-based deep learning pipeline builder',
  favicon: 'img/favicon.ico',

  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Deployed to the GitHub Pages project URL:
  //   https://treeleaves30760.github.io/CodefyUI/
  // To switch to a custom subdomain later:
  //   1. set `url` to 'https://your-subdomain' and `baseUrl` to '/'
  //   2. add a `static/CNAME` file containing your subdomain (one line)
  //   3. set the custom domain in repo Settings → Pages
  url: 'https://treeleaves30760.github.io',
  baseUrl: '/CodefyUI/',

  // GitHub Pages deployment config.
  organizationName: 'treeleaves30760',
  projectName: 'CodefyUI',
  trailingSlash: false,

  onBrokenLinks: 'throw',

  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'zh-TW'],
    localeConfigs: {
      en: {label: 'English'},
      'zh-TW': {label: '繁體中文', htmlLang: 'zh-Hant-TW'},
    },
  },

  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/', // docs served at the site root
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/treeleaves30760/CodefyUI/tree/main/docs/',
          editLocalizedFiles: true,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themes: [
    [
      require.resolve('@easyops-cn/docusaurus-search-local'),
      {
        hashed: true,
        language: ['en', 'zh'],
        indexDocs: true,
        indexBlog: false,
        docsRouteBasePath: '/',
      },
    ],
  ],

  themeConfig: {
    image: 'img/codefyui-social-card.png',
    colorMode: {
      defaultMode: 'dark',
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'CodefyUI',
      logo: {
        alt: 'CodefyUI Logo',
        src: 'img/logo.png',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          type: 'localeDropdown',
          position: 'right',
        },
        {
          href: 'https://github.com/treeleaves30760/CodefyUI',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {label: 'Introduction', to: '/'},
            {label: 'Installation', to: '/getting-started/installation'},
            {label: 'Teaching Inspector', to: '/usage/teaching-inspector'},
            {label: 'Plugins', to: '/advanced/plugins'},
          ],
        },
        {
          title: 'Project',
          items: [
            {label: 'GitHub', href: 'https://github.com/treeleaves30760/CodefyUI'},
            {label: 'Issues', href: 'https://github.com/treeleaves30760/CodefyUI/issues'},
            {
              label: 'Plugin Template',
              href: 'https://github.com/treeleaves30760/CodefyUI-Plugin-Official',
            },
          ],
        },
        {
          title: 'License',
          items: [
            {
              label: 'AGPL-3.0',
              href: 'https://github.com/treeleaves30760/CodefyUI/blob/main/LICENSE',
            },
            {
              label: 'Commercial',
              href: 'https://github.com/treeleaves30760/CodefyUI/blob/main/COMMERCIAL-LICENSE.md',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} CodefyUI. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'python', 'json', 'powershell', 'toml'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
