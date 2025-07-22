import React from 'react'
import Document, { Html, Head, Main, NextScript } from 'next/document'

class MyDocument extends Document {
  render() {
    return (
      <Html lang="en">
        <Head>
          {/* Basic meta tags */}
          <meta charSet="utf-8" />
          <meta name="description" content="SkyPilot Dashboard - Run AI on Any Infra. Unified, Faster, Cheaper. Manage your clusters, jobs, and infrastructure from a single interface." />
          <meta name="keywords" content="SkyPilot, AI, cloud, infrastructure, clusters, jobs, GPU, machine learning, MLOps" />
          <meta name="author" content="SkyPilot Team" />
          
          {/* Open Graph / Facebook */}
          <meta property="og:type" content="website" />
          <meta property="og:url" content="https://docs.skypilot.co/" />
          <meta property="og:title" content="SkyPilot Dashboard - Run AI on Any Infra" />
          <meta property="og:description" content="SkyPilot Dashboard - Run AI on Any Infra. Unified, Faster, Cheaper. Manage your clusters, jobs, and infrastructure from a single interface." />
          <meta property="og:image" content="/dashboard/skypilot-thumbnail.png" />
          <meta property="og:image:alt" content="SkyPilot Logo" />
          <meta property="og:image:width" content="1000" />
          <meta property="og:image:height" content="219" />
          <meta property="og:site_name" content="SkyPilot" />
          
          {/* Twitter */}
          <meta property="twitter:card" content="summary_large_image" />
          <meta property="twitter:url" content="https://docs.skypilot.co/" />
          <meta property="twitter:title" content="SkyPilot Dashboard - Run AI on Any Infra" />
          <meta property="twitter:description" content="SkyPilot Dashboard - Run AI on Any Infra. Unified, Faster, Cheaper. Manage your clusters, jobs, and infrastructure from a single interface." />
          <meta property="twitter:image" content="/dashboard/skypilot-thumbnail.png" />
          <meta property="twitter:image:alt" content="SkyPilot Logo" />
          <meta property="twitter:site" content="@skypilot_org" />
          <meta property="twitter:creator" content="@skypilot_org" />
          
          {/* Additional meta tags */}
          <meta name="theme-color" content="#372F8A" />
          <meta name="apple-mobile-web-app-capable" content="yes" />
          <meta name="apple-mobile-web-app-status-bar-style" content="default" />
          <meta name="apple-mobile-web-app-title" content="SkyPilot Dashboard" />
          
          {/* Favicon */}
          <link rel="icon" href="/dashboard/favicon.ico" />
          <link rel="apple-touch-icon" href="/dashboard/skypilot.svg" />
          
          {/* Canonical URL */}
          <link rel="canonical" href="https://docs.skypilot.co/" />
        </Head>
        <body>
          <Main />
          <NextScript />
        </body>
      </Html>
    )
  }
}

export default MyDocument 
