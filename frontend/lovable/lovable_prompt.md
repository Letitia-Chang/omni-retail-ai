# OmniRetail AI Dashboard

Build a modern AI-powered retail marketing dashboard.

The system helps marketing teams:

- analyze customer segments
- view campaign recommendations
- explore product strategies
- generate AI-powered campaign ideas

The frontend should connect to a FastAPI backend.

Backend API:

- GET /segments
- GET /campaigns
- GET /campaigns/{segment}
- GET /summary

Design requirements:

- modern SaaS dashboard aesthetic
- dark/light mode support
- responsive layout
- clean data visualization
- cards, tables, and charts
- AI copilot feel

Pages to build:

## 1. Dashboard Overview

Show:

- total campaign recommendations
- total customer segments
- average campaign score
- summary charts

Charts:

- campaign score distribution
- top customer segments
- promotion strategy counts

## 2. Customer Segment Explorer

Sidebar:

- selectable customer segments

Main panel:

- top recommended products
- campaign score
- promotion strategy
- ranking explanation

Display:

- product cards
- inventory level
- purchase probability
- target audience
- copy angles

## 3. Campaign Recommendation Table

Interactive searchable table with:

- product
- segment
- campaign score
- promotion strategy
- recommendation explanation

Include:

- sorting
- filtering
- pagination

## 4. AI Campaign Generator

Display:

- generated campaign messages
- copy angles
- recommended marketing strategy

Allow:

- regenerate campaign copy
- select customer segment
- select product

## 5. Analytics Page

Charts:

- average purchase probability by segment
- inventory distribution
- campaign score heatmap
- segment-level recommendation performance

Frontend stack:

- React
- Tailwind CSS
- shadcn/ui
- Recharts

Design style:

- modern AI SaaS platform
- similar to HubSpot + Salesforce + Notion AI
- minimal but polished

Color palette:

- white / neutral backgrounds
- teal or emerald accent
- soft shadows
- rounded cards

Important:

- all data comes from FastAPI endpoints
- use loading states
- use reusable components
- keep architecture clean
