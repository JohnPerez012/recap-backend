  /**
 * RE-CAPS Staff Copilot Knowledge Base Ingestion Script
 * Populates Pinecone index: recaps-staff-copilot
 * Embedding Model: multilingual-e5-large (1024 dimensions) via Pinecone Inference API
 *
 * Supports Role-Based Metadata Scoping:
 * - role_scope: ['admin'] (Admin operational governance across all 9 sidebar areas + AI Copilot)
 * - role_scope: ['librarian'] (Librarian portal workflows & boundaries)
 * - role_scope: ['admin', 'librarian', 'shared'] (Shared physical cataloging standards)
 */

const path = require('path');
require('dotenv').config({ path: path.resolve(__dirname, '.env') });

const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_HOST = process.env.PINECONE_STAFF_COPILOT_HOST || 'recaps-staff-copilot-wf1edxk.svc.aped-4627-b74a.pinecone.io';
const DEPRECATED_IDS = ['admin_maintenance_mode_activation'];

if (!PINECONE_API_KEY) {
  console.error('ERROR: PINECONE_API_KEY is not defined in recap-backend/.env');
  process.exit(1);
}

// -------------------------------------------------------------
// Knowledge Chunks Definition (Each chunk kept < 1900 chars for 512-token E5 safety)
// -------------------------------------------------------------
const KNOWLEDGE_CHUNKS = [
  // =============================================================
  // 1. DASHBOARD & NAVIGATION & SESSION LIFECYCLE
  // =============================================================
  {
    id: 'admin_dashboard_overview_kpis',
    role_scope: ['admin'],
    category: 'dashboard_overview',
    title: 'Admin Dashboard Overview, Live KPI Cards, and Recent Feeds',
    landmark: 'Sidebar Rail > "Dashboard" > Live Updates Header & KPI Counter Cards',
    text: `Admin Dashboard Overview (#section-dashboard):
[Area: Dashboard Section > Top Status Bar & KPI Grid] > [Element: "Refresh" Button (#refresh-dashboard-btn) & Interactive KPI Cards]
1. Click "Dashboard" at the top of the left Sidebar Rail (#sidebar-rail).
2. View the header showing "Dashboard Overview", the "Live Updates" pulse indicator, #dash-last-updated timestamp, and the "Refresh" button (#refresh-dashboard-btn).
3. Click any of the 5 live KPI cards to jump directly to its section:
   - Total Projects (#dash-stat-projects) opens Projects.
   - Pending Approvals (#pending-approvals-card) opens Approvals.
   - Total Users (#dash-stat-users) and Role Distribution (#dash-stat-roles showing #stat-students-count and #stat-librarians-count) open Users.
   - Recent Activities (#dash-stat-activity) opens Activity.
4. Use "Recent Projects" (#recent-projects-list, Latest 5 + "View All") to open a project preview modal directly, or "Recent Users" (#recent-users-list, Latest 5 + "View All") to jump to Users filtered to that account.
Precaution: Dashboard KPIs reflect cached/live Firestore state; click "Refresh" (#refresh-dashboard-btn) if numbers appear stale.`
  },
  {
    id: 'admin_navigation_home_and_theme',
    role_scope: ['admin'],
    category: 'navigation',
    title: 'Sidebar Navigation, Returning Home, Theme Switcher, and Mobile Drawer',
    landmark: 'Sidebar Rail Footer > "Home" (#back-to-home-btn) & Topbar Theme Switcher',
    text: `Navigating Between Admin Sections, Public Home Portal, and Theme Modes:
[Area: Sidebar Rail > Footer Navigation] > [Element: "Home" Link (#back-to-home-btn)]
1. The Admin Sidebar Rail contains 8 main sections (Dashboard, Projects, Approvals, Users, Analytics, Copilot, Activity, Settings) plus 2 footer actions: "Home" (#back-to-home-btn) and "Logout" (#admin-logout-btn).
2. Click "Home" (#back-to-home-btn) in the lower-left sidebar footer to navigate back to the public search catalog (../index.html). Going Home preserves your active Admin login session and saves the closed state of the Copilot widget.
3. Click the topbar theme toggle button (#theme-toggle-admin) to switch between dark and light surface themes cleanly.
4. On tablet or mobile viewports (width <= 1024px), tap the top-left menu button (#mobile-rail-toggle) to open the slide-over navigation drawer; click the backdrop overlay (#rail-overlay) or press Esc to close it.
Precaution: Clicking "Home" does NOT log you out. Use the red "Logout" button (#admin-logout-btn) if ending your Admin session on a shared computer.`
  },
  {
    id: 'admin_logout_session_workflow',
    role_scope: ['admin'],
    category: 'authentication',
    title: 'Signing Out Securely via Admin Logout Confirmation Modal',
    landmark: 'Sidebar Rail Footer > "Logout" (#admin-logout-btn) > Logout Confirmation Modal',
    text: `How Administrator Logs Out Securely:
[Area: Sidebar Rail > Footer Navigation] > [Element: "Logout" Button (#admin-logout-btn)]
1. Click the red "Logout" item (#admin-logout-btn) at the bottom of the left Sidebar Rail.
2. A shared confirmation modal (LogoutModal) opens with "Cancel" and "Logout" buttons so accidental clicks do not end your session.
3. Click "Logout" in the modal to confirm sign-out.
4. RE-CAPS records a local logout event, runs sessionStorage.clear(), clears localStorage key cachedAuthState, executes firebase.auth().signOut(), and redirects to ../index.html.
Precaution: Finish saving any open Project Modal draft or active Pinecone sync before logging out, because confirming logout immediately clears session state.`
  },

  // =============================================================
  // 2. PROJECTS MANAGEMENT, MODALS, BULK IMPORT & PINECONE SYNC
  // =============================================================
  {
    id: 'admin_view_browse_existing_projects',
    role_scope: ['admin'],
    category: 'projects_cataloging',
    title: 'Browsing, Searching, Sorting, and Filtering Cataloged Projects',
    landmark: 'Projects Section > Search (#projects-search) & Filter Pills (#project-filters)',
    text: `Viewing, Searching, and Filtering Existing Projects (#section-projects):
[Area: Projects Section > Filter Pills & All Projects Table] > [Element: Search Input (#projects-search) & Program Filter Pills]
1. Click "Projects" in the left Sidebar Rail.
2. Search via #projects-search across title, authors, abstract, topics, keywords, adviser, program, year, and shelf coordinates; matches display "Matched in" chips under the title.
3. Toggle multi-select filter pills (#project-filters): "All Projects", 10 campus programs (BSFi, BSIT, BSHM, BSIE, BSEd-Math, BEEd, BIT-Computer, BIT-Electronics, BIT-Automotive, BTLEd-HE), "Recent (30 days)", and "Unsynced" (#filter-unsynced).
4. Sort columns by clicking headers: Title, Authors (whitelist-copyable under R.A. 10173), Program, Year, Adviser, or inspect sync dots beside titles (clicking triggers single-row syncSingleProject).
5. Use "Refresh" (#refresh-projects-btn) to pull live Firestore updates.
Precaution: Titles and Abstracts are protected against copying under R.A. 10173, whereas Authors remain copyable for academic citation.`
  },
  {
    id: 'admin_add_new_project_procedure',
    role_scope: ['admin'],
    category: 'projects_cataloging',
    title: 'Adding a New Capstone Project Directly as Administrator',
    landmark: 'Projects Section > Table Header Action Bar > "Add Project" Button (#add-project-btn)',
    text: `Adding a Capstone Project Directly (#project-modal):
[Area: Projects Section > Table Header Action Bar] > [Element: Emerald "Add Project" Button (#add-project-btn)]
1. Open "Projects" and click "Add Project" (#add-project-btn, shortcut Ctrl+Shift+P). Optional Auto-save toggle (#project-autosave-toggle) saves drafts locally in admin_project_draft.
2. Section 01 (Core Identity): Enter unique Project Title (checked via titleNorm on save) and add at least 1 author using #add-author-btn.
3. Section 02 (Academic Details): Enter Adviser, Academic Year (readonly spinner 2023 to 2100), select one of 10 Programs (BSFi, BSIT, BSHM, BSIE, BSEd-Math, BEEd, BIT-Computer, BIT-Electronics, BIT-Automotive, BTLEd-HE), and pick Status (Completed, Published, or Draft).
4. Section 03 (Research Content): Enter Abstract plus individual Topic and Keyword rows.
5. Section 04 (Book & Representation Images): Optionally upload PNG/JPG/WEBP images (max 10 MB each, Cloudinary storage, rotate/mirror controls); photos are optional for Admin saves.
6. Section 05 (Physical Catalog & Librarian Accountability): Optionally add Shelf Location, Call Number, and Accession Number alongside #admin-catalog-audit-box.
7. Click "Save Project" (#submit-project-btn).
Precaution: Admin projects save directly to Firestore projects without Approvals queue and auto-sync to Pinecone index recaps-projects-search.`
  },
  {
    id: 'admin_edit_existing_project',
    role_scope: ['admin'],
    category: 'projects_cataloging',
    title: 'Editing an Existing Project and Handling Active Librarian Locks',
    landmark: 'Projects Section > All Projects Table > Actions Column > "Edit" Button',
    text: `Updating an Existing Project Record:
[Area: Projects Section > All Projects Table] > [Element: "Edit" Button in Row Actions Column]
1. Open "Projects" and locate the target row via #projects-search or #project-filters.
2. Rows are unlocked by default showing View, Edit, Delete, and Lock. If a row shows "Being edited/cataloged" by a Librarian, wait for the 45-second lease to expire or click the force-release "x" icon (__adminForceReleaseCatalogLock). If previously Admin-locked, click "Unlock" first.
3. Click "Edit" to open #project-modal populated with Sections 01 through 05 (including #admin-catalog-audit-box accountability metadata).
4. Update Title, Authors, Adviser, Year, Program, Status (Completed, Published, Draft), Abstract, Topics, Keywords, Section 04 Images (rotate/mirror), or Section 05 Shelf/Call/Accession details.
5. Click "Save Project" (#submit-project-btn). Duplicate titleNorm collisions are blocked automatically.
Precaution: Editing searchable fields (title, abstract, authors, topics, keywords, program, year) automatically re-embeds and updates the vector in Pinecone index recaps-projects-search.`
  },
  {
    id: 'admin_unlock_delete_project',
    role_scope: ['admin'],
    category: 'catalog_security',
    title: 'Locking, Unlocking, and Securely Deleting a Project (#secure-delete-modal)',
    landmark: 'Projects Section > All Projects Table > Actions Column > "Delete" & "Lock/Unlock"',
    text: `Locking, Unlocking, and Permanently Deleting a Project:
[Area: Projects Section > All Projects Table] > [Element: Actions Column ("Delete", "Lock", "Unlock")]
1. By default, Admin project rows are UNLOCKED and immediately display View, Edit, Delete, and Lock. Clicking "Lock" protects a row (hiding Edit and Delete until Admin clicks "Unlock").
2. To delete an unlocked project, click "Delete" in its row Actions column to open #secure-delete-modal.
3. Read the modal warning carefully and type the EXACT project title into the confirmation input. (RE-CAPS does NOT use any secret verification code, PIN, or OTP for project deletion.)
4. Click "Delete" in #secure-delete-modal to confirm.
5. Confirmation deletes the Firestore projects document, prunes its Cloudinary images, decrements the RTDB project counter, and removes the vector from Pinecone index recaps-projects-search.
Precaution: Project deletion is irreversible and permanently removes metadata, Cloudinary assets, and Pinecone search vectors.`
  },
  {
    id: 'admin_bulk_import_excel',
    role_scope: ['admin'],
    category: 'bulk_import',
    title: 'Bulk Importing Multiple Projects via Excel or JSON (#bulk-import-modal)',
    landmark: 'Projects Section > Table Header Action Bar > "Bulk Import" (#bulk-import-btn)',
    text: `Bulk Importing Projects from Excel (.xlsx/.xls) or JSON (#bulk-import-modal):
[Area: Projects Section > Table Header Action Bar] > [Element: "Bulk Import" Button (#bulk-import-btn)]
1. Open "Projects" and click "Bulk Import" (#bulk-import-btn).
2. Download either the "Excel Template (.xlsx)" or "JSON Template (.json)" inside #bulk-import-modal.
3. Prepare rows using ONLY the 9 supported columns: 4 required (title, authors, program, year) and 5 optional (adviser, abstract, keywords, topics, status which defaults to Completed). Bulk Import has NO shelf location, callNumber, or accessionNo columns.
4. Upload one or multiple .xlsx/.xls/.json files to stage multi-batch imports and optionally attach per-project book images in the staging tray.
5. Review parsed validation feedback (existing duplicate titleNorm entries are automatically skipped) and click "Import Projects".
6. Imported records write to Firestore and trigger automatic Pinecone syncing; add physical shelf fields later via Edit > Section 05.
Precaution: Separate multiple authors, keywords, or topics cleanly according to template examples and keep year within 2023-2100.`
  },
  {
    id: 'admin_pinecone_sync_procedure',
    role_scope: ['admin'],
    category: 'pinecone_sync',
    title: 'Batch and Single-Row Syncing to Pinecone Index (recaps-projects-search)',
    landmark: 'Projects Section > Table Header Action Bar > "Sync Pinecone" (#sync-pinecone-btn)',
    text: `Syncing Capstone Projects to Pinecone Vector Search:
[Area: Projects Section > Table Header Action Bar] > [Element: "Sync Pinecone" Button (#sync-pinecone-btn)]
1. Open "Projects" and click "Sync Pinecone" (#sync-pinecone-btn) to open #pinecone-sync-modal, which displays "Index: recaps-projects-search" plus Total, Indexed, and Unsynced counts.
2. Select sync mode: "Sync Only Unsynced Projects" (faster delta sync) or "Full Index Re-Sync (All Projects)".
3. Click "Start Pinecone Sync" (#pinecone-sync-confirm-btn). Closing the modal is intentionally locked while sync runs.
4. When finished, inspect the result summary; if any record fails, click "Retry Failed" (#pinecone-sync-retry-failed-btn).
5. You can also sync a single row directly by clicking the sync status dot beside that project's title (syncSingleProject), and "Save Project", "Approve", and "Import Projects" sync automatically.
Precaution: Do not close or refresh the browser tab while a batch sync job is in progress.`
  },
  {
    id: 'admin_troubleshoot_rag_catalog',
    role_scope: ['admin'],
    category: 'pinecone_sync',
    title: 'Troubleshooting Theses Missing from Public Search or Student AI Chatbot',
    landmark: 'Projects Section > Filter Pills > "Unsynced" Pill (#filter-unsynced)',
    text: `Diagnosing Missing Theses in Public Catalog or Student RAG Chatbot:
[Area: Projects Section > Filter Pills] > [Element: "Unsynced" Filter Pill (#filter-unsynced)]
1. First check "Approvals" in the Sidebar Rail: submissions created by Librarians stay in pendingProjects until an Admin clicks "Evaluate" > "Approve & Publish Publicly" (sets status to Completed and isPublic to true).
2. Next open "Projects" and verify the project's Status in Edit > Section 02 is Completed or Published (Draft entries are excluded from public AI search).
3. Click the "Unsynced" filter pill (#filter-unsynced) to list records lacking a synced vector in recaps-projects-search.
4. Click the sync status dot beside the project title for single-row sync, or open "Sync Pinecone" (#sync-pinecone-btn) and run "Sync Only Unsynced Projects" or "Full Index Re-Sync".
Precaution: Public semantic RAG search requires both a published Firestore record and an indexed 1024-dim vector in recaps-projects-search.`
  },

  // =============================================================
  // 3. APPROVALS & DUPLICATE CHECK WORKFLOW
  // =============================================================
  {
    id: 'admin_approvals_evaluate_procedure',
    role_scope: ['admin'],
    category: 'approvals',
    title: 'Evaluating, Batch-Approving, and Checking Duplicates for Librarian Submissions',
    landmark: 'Approvals Section > Toolbar ("Approve All" #batch-approve-all-btn) & Submissions Table ("Evaluate")',
    text: `Evaluating and Publishing Librarian Submissions (#section-approvals):
[Area: Approvals Section > Pending Submissions Queue] > [Element: "Evaluate" Button OR "Approve All" (#batch-approve-all-btn)]
1. Open "Approvals" in the Sidebar Rail (badge counter #approvals-live-counter, shortcut Ctrl+Shift+A for batch approve). Filter by #approvals-search, #approvals-program-filter, or click "Refresh" (#refresh-approvals-btn).
2. If a row displays "Being edited by: [Name]", wait until the librarian closes their draft editor before acting.
3. Click "Evaluate" to open #approval-review-modal. Admin can directly edit Title, Authors (#approval-add-author-btn), Program, Year, Adviser, Keywords, Abstract, and inspect submitted photos before deciding.
4. Click "Approve & Publish Publicly" (or "Reject & Discard" to remove). For multiple items, click "Approve All" (#batch-approve-all-btn).
5. Both single and batch approvals run duplicate checks (findPotentialDuplicates) against projects and pendingProjects; any similarity triggers #approval-duplicate-confirm-modal showing VS cards with "Evaluate & Edit Details", "Approve Clean Only", or "Approve Anyway".
6. Publishing writes to projects (status: Completed, isPublic: true), deletes pendingProjects/{id}, and syncs to Pinecone.
Precaution: Approved projects become immediately public; inspect duplicate warnings before choosing "Approve Anyway".`
  },

  // =============================================================
  // 4. USERS GOVERNANCE, LIBRARIAN CREATION & QUARANTINE PURGE
  // =============================================================
  {
    id: 'admin_user_account_management',
    role_scope: ['admin'],
    category: 'user_management',
    title: 'Searching Users, Filter Pills, Statuses, and Available Row Actions',
    landmark: 'Users Section > Search (#users-search), 6 Filter Pills (#user-filters) & Actions Column',
    text: `Managing User Accounts in Users Section (#section-users):
[Area: Users Section > Search & Filter Pills] > [Element: #users-search & #user-filters]
1. Click "Users" in the Sidebar Rail to view columns: User, Email, User Type, Account Status, Joined, Last Login, and Actions.
2. Search by name, email, or role using #users-search (searching "faculty" automatically matches teacher accounts).
3. Select any of the 6 Filter Pills in #user-filters: "All Users", "Admins", "Librarians", "Students", "Suspended" (with #suspended-count-badge), and "Quarantine (14d Trash)" (with #quarantine-count-badge). Note: there is NO "Faculty" pill.
4. Available row actions depend strictly on account state:
   - Admin rows: NO Sanction or Delete buttons (protected against self-lockout).
   - Active non-Admin rows: "Sanction" button (opens #user-governance-modal).
   - Suspended rows: "Reactivate" and "Sanction".
   - Quarantined rows: "Restore" and "Purge".
Precaution: The Users table does NOT contain Key (password reset), Pencil (edit role), or Unlock/Shield buttons; account enforcement goes strictly through Sanction / Reactivate / Restore / Purge.`
  },
  {
    id: 'admin_delete_user_account',
    role_scope: ['admin'],
    category: 'user_management',
    title: 'Suspending, Quarantining (14d Trash), Restoring, and Purging User Accounts',
    landmark: 'Users Section > Actions Column > "Sanction" (#user-governance-modal) / "Purge"',
    text: `Sanctioning, Soft-Quarantining, and Permanently Purging a User Account:
[Area: Users Section > Users Table] > [Element: "Sanction" -> #user-governance-modal & "Purge" in Quarantine]
1. In "Users", find an active non-Admin user row and click "Sanction" to open the 3-tier Governance Modal (#user-governance-modal).
2. Tier 1 (Enforcement Action): Select either "Account Suspension (Freeze)" with duration (24h, 3d, 7d, 30d, or indefinite) OR "14-Day Deletion Quarantine" (soft-delete).
3. Tier 2 (Reason & Justification): Choose a Violation Category and enter at least 15 characters of justification notes.
4. Tier 3 (Affirmation): Check the affirmation checkbox and submit.
5. To reverse or finalize deletion, click the "Quarantine (14d Trash)" pill: click "Restore" to return the account to active status, or click "Purge" to call DELETE /api/users/:uid, permanently removing Firebase Auth and Firestore user records.
Precaution: Purging permanently deletes login credentials while preserving historical authored or advised capstone projects. Administrator accounts cannot be sanctioned or purged.`
  },
  {
    id: 'admin_create_librarian_staff',
    role_scope: ['admin'],
    category: 'user_management',
    title: 'Creating a Librarian Staff Account (#create-librarian-modal)',
    landmark: 'Users Section > Table Header > "Create Librarian" Button (#create-librarian-btn)',
    text: `Registering a New Librarian Staff Account:
[Area: Users Section > Header Action Toolbar] > [Element: "Create Librarian" Button (#create-librarian-btn)]
1. Open "Users" in the Sidebar Rail and click "Create Librarian" (#create-librarian-btn) to open #create-librarian-modal.
2. Fill in Full Name, institutional Email prefix (automatically formatted with @ctu.edu.ph suffix chip), and Position/Designation (datalist options defaulting to "Librarian").
3. Enter a Password of at least 6 characters (or click "Auto-Generate" beside the field to generate a strong password with live strength meter) and Confirm Password.
4. Click "Create Librarian" to provision the account via POST /api/users/create-librarian (with secondary Firebase Auth client fallback if needed).
5. On success, copy the displayed credentials using "Copy Credentials" and provide the Librarian portal link (/pages/login.html).
Precaution: Newly created Librarian accounts can only draft thesis submissions into pendingProjects; Admin approval is required before public indexing.`
  },

  // =============================================================
  // 5. ANALYTICS, DRILLDOWNS, EXPORTS & LOCAL ACTIVITY LOGS
  // =============================================================
  {
    id: 'admin_print_accreditation_report',
    role_scope: ['admin'],
    category: 'reports',
    title: 'Printing Institutional Accreditation Reports and Exporting Filtered CSV',
    landmark: 'Analytics Section > Command Toolbar > "Print Report" (#analytics-print-btn) & "Export CSV" (#analytics-export-btn)',
    text: `Generating Printable Reports and CSV Exports in Analytics (#section-analytics):
[Area: Analytics Section > Command Toolbar] > [Element: "Print Report" (#analytics-print-btn) & "Export CSV" (#analytics-export-btn)]
1. Click "Analytics" in the Sidebar Rail.
2. Scope your report using the Program dropdown (#analytics-program-filter) and Time Horizon tabs ("All Time", "This Year", "90 Days", "30 Days"). Use "Reset Filter" (#analytics-reset-filters-btn) to clear filters or "Refresh" (#analytics-refresh-btn) to reload live Firestore data.
3. Click "Print Report" (#analytics-print-btn): RE-CAPS populates #analytics-print-header with filter scope and timestamp (#analytics-print-timestamp), then triggers window.print() for official Dean / CHED hardcopy or Save-as-PDF output.
4. Click "Export CSV" (#analytics-export-btn) to download a UTF-8 BOM Excel-friendly CSV file containing the filtered analytics dataset.
Precaution: Always verify your Program filter and Time Horizon selection before clicking Print Report or Export CSV so figures match the requested accreditation window.`
  },
  {
    id: 'admin_system_audit_logs_analytics',
    role_scope: ['admin'],
    category: 'system_audit',
    title: 'Analytics KPIs/Chart Drilldowns & Local Browser Activity Trail',
    landmark: 'Analytics Section (#section-analytics) & Activity Section (#section-activity)',
    text: `Analytics Charts, Drilldowns, and Activity History:
[Area: Analytics Section & Activity Section] > [Element: Interactive Charts (#section-analytics) & Local Activity Controls (#section-activity)]
1. Analytics (#section-analytics) shows 6 KPIs (Total Capstones, Repository Community, Thesis Bookmarks, Active Programs, Intake Last 30d, AI Vector Index % Synced), 4 Chart.js visualizations (Program Distribution Bar/Horizontal/Donut, User Roles Doughnut, Intake Timeline Cumulative/Annual, Top Advisers Horizontal Bar), and Top 10 Bookmarked Theses table ("View"). Clicking slices/bars in Program Distribution drills directly into Projects; clicking User Roles drills into Users.
2. Activity (#section-activity) displays Admin actions stored strictly in browser localStorage (key recaps_activities_<uid>, capped at 200 items, zero Firestore writes to stay on Spark free tier). Controls: Refresh, Export CSV/JSON, Clear History, Bulk Select (Select All, Export Selected, Remove Selected), Date filter (All Time/Today/7d/30d), 9 Category pills, Storage Used + Device summary, and Details modal (Delete Record, View Manuscript).
Precaution: Activity logs are local to your browser profile and do NOT track IP addresses or server-wide logs.`
  },

  // =============================================================
  // 6. SETTINGS, SECURITY, CACHE/EXPORT & AI COPILOT BOUNDARIES
  // =============================================================
  {
    id: 'admin_settings_profile_and_security',
    role_scope: ['admin'],
    category: 'system_settings',
    title: 'Settings Profile Card, Profile Editing, and 6-Layer Telegram OTP Password Change',
    landmark: 'Settings Section > Profile Card & Security Card ("Edit Profile Info" / "Change Password")',
    text: `Managing Administrator Profile & Password Security (#section-settings):
[Area: Settings Section > Profile & Security Cards] > [Element: "Edit Profile Info" (#update-profile-btn) & "Change Password" (#change-password-btn)]
1. Open "Settings" in the Sidebar Rail. The sticky left Profile Card shows Copy Email (#aset-card-copy-email-btn) and quick stat buttons jumping to Holdings (#aset-btn-holdings -> Projects), Pending (#aset-btn-pending -> Approvals), and Users (#aset-btn-users).
2. Click "Edit Profile Info" (#update-profile-btn) to open #admin-profile-modal and update Full Name. Institutional @ctu.edu.ph email and "University Library" department are readonly.
3. Click "Change Password" (#change-password-btn) to start the 6-layer change password flow: enter Current Password AND complete 6-digit Telegram OTP verification via @RecapsOtpBot (#send-otp-btn / #link-chat-id-input).
4. New passwords must be 8-128 characters with uppercase, lowercase, number, and special character. Security enforces 10 changes per 24h, locks after 5 OTP/password failures for 1h, revokes refresh tokens, and sends Telegram + Resend alerts.
Precaution: Keep your Telegram OTP bot chat linked before attempting a password rotation.`
  },
  {
    id: 'admin_settings_cache_export_diagnostics',
    role_scope: ['admin'],
    category: 'system_settings',
    title: 'Settings Cache Purge, Catalog Export (CSV/JSON), Diagnostics & Keyboard Shortcuts',
    landmark: 'Settings Section > Data Management Card (#settings-card-maintenance) & Shortcuts Card',
    text: `Data Integrity Utilities, Diagnostics, and Shortcuts in Settings (#section-settings):
[Area: Settings Section > Data Management & Diagnostics] > [Element: "Purge Cache" (#btn-purge-cache) & "CSV"/"JSON" Export]
1. In "Settings", scroll to #settings-card-maintenance ("Data Management & Cache Utilities").
2. Click "Purge Cache" (#btn-purge-cache) to clear cached dashboard data and reload fresh state.
3. Under "Export Repository Catalog", click "CSV" (#btn-export-catalog-csv) or "JSON" (#btn-export-catalog-json) to download a full catalog backup.
4. Review the 4-cell System Diagnostics grid for runtime metrics, and inspect the Shortcuts Card:
   - Ctrl+K: Focus active search
   - Ctrl+Shift+P: Open Add Project modal
   - Ctrl+Shift+A: Switch to Approvals section (does NOT trigger batch approve)
   - Ctrl+Shift+C: Toggle AI Copilot
   - Esc: Close open modal or mobile drawer
Precaution: Settings contains ONLY Profile, Telegram OTP Password Change, Purge Cache, Catalog CSV/JSON Export, Diagnostics, and Shortcuts — there are no Maintenance Mode or Registration toggles.`
  },
  {
    id: 'admin_ai_copilot_usage_and_limits',
    role_scope: ['admin'],
    category: 'copilot_governance',
    title: 'Using the Admin AI Copilot Workspace, Widget, and Read-Only Boundaries',
    landmark: 'Sidebar Rail > "Copilot" (#rail-nav-copilot) OR Floating Widget (#admin-floating-copilot-btn)',
    text: `Operating the Admin AI Copilot & Understanding Read-Only Boundaries:
[Area: AI Copilot Workspace (#section-copilot) OR Floating Copilot Widget (#admin-copilot-widget)] > [Element: Copilot Input & Quick Prompt Chips]
1. Access Copilot via "Copilot" in the Sidebar Rail (#rail-nav-copilot), the bottom-right floating button (#admin-floating-copilot-btn), or shortcut Ctrl+Shift+C.
2. In the floating widget, click #copilot-widget-fullscreen-btn to expand into #section-copilot, click quick prompt chips for common workflows, or click "Copy Guidance" and Quick-Jump action buttons inside any answer.
3. Strict Advisory Boundary: Admin AI Copilot is strictly READ-ONLY. It guides exact screen locations and navigates you to sections/buttons, but CANNOT directly create/edit/delete Firestore records, approve/reject submissions, sync Pinecone, sanction/purge users, or change passwords.
Precaution: Always execute actual data mutations yourself using the linked section modal or confirmation button.`
  },
  {
    id: 'admin_system_negative_boundaries',
    role_scope: ['admin'],
    category: 'anti_hallucination',
    title: 'Non-Existent & Legacy Features Guard (Anti-Hallucination Reference)',
    landmark: 'Admin Portal Verified Feature Inventory',
    text: `Verified Feature Boundaries — Actions and Controls That DO NOT Exist in Admin UI:
[Area: Admin Portal System Boundaries] > [Element: Verified Feature Constraints]
1. NO Maintenance Mode & NO Allow Registration / Guest Browsing toggles exist anywhere in Settings (#section-settings).
2. NO Key icon (password reset), NO Pencil icon (role edit), NO Shield/Unlock button, and NO instant red Trash delete icon exist in the Users table (#section-users). Use "Sanction" (#user-governance-modal), "Reactivate", "Restore", or "Purge" in "Quarantine (14d Trash)". Also there is NO "Faculty" filter pill (type "faculty" in #users-search to filter teacher accounts).
3. NO secret verification code, PIN, or OTP is used to delete a project — #secure-delete-modal requires typing the exact project title.
4. NO locked-by-default rows exist on the Projects table; rows start unlocked with View, Edit, Delete, and Lock visible (unless actively locked by a Librarian or manually Admin-locked).
5. NO location, callNumber, or accessionNo columns exist in Bulk Import Excel/JSON templates (9 metadata columns only).
6. NO IP address logs or Firestore activity audit collection exist in Activity (#section-activity); logs are browser-local only.
7. NO topbar theme toggle button (#theme-toggle-admin) is active on the Admin Dashboard (surface theme is inherited automatically from localStorage/OS preference and shown in Settings Diagnostics), and Ctrl+Shift+A switches to Approvals rather than batch-approving.
Precaution: If asked about any removed or legacy control above, clearly state it does not exist in the current Admin interface and point to the actual supported workflow.`
  },

  // =============================================================
  // 7. SHARED PHYSICAL CATALOGING STANDARDS (ADMIN SECTION 05 & LIBRARIAN STUDIO)
  // =============================================================
  {
    id: 'shelf_location_format',
    role_scope: ['admin', 'librarian', 'shared'],
    category: 'shelf_cataloging',
    title: 'Physical Shelf Location Formatting Rules',
    landmark: 'Admin Project Modal (#project-modal) > Section 05 (Physical Catalog & Librarian Accountability) / Librarian Physical Cataloging Studio > Physical Location Note',
    text: `How to Format Physical Shelf Location in RE-CAPS:
For Admins: Open Section 05 (Physical Catalog & Librarian Accountability) inside #project-modal and locate the Physical Location Note input (#project-catalog-location). Note: Section 04 is Book & Representation Images, not shelf location.
For Librarians: Open the Physical Cataloging Studio modal and expand Physical Location Note (#cn-sec-location > #cn-note).
Standard Format: "Rack [X], Row [Y], Stand [Z], Color [Bind Color]" (Example: "Rack 1, Row 2, Stand 3, Color Blue").
Include the Shelf/Rack number, vertical Row, horizontal Stand, and physical book cover/spine binding color so students viewing project details or using the student chatbot can locate bound manuscripts inside CTU Daanbantayan Library.
Precaution: Keep shelf coordinates consistent when cataloging or editing bound theses so physical holdings match digital records.`
  },
  {
    id: 'call_number_dewey_standards',
    role_scope: ['admin', 'librarian', 'shared'],
    category: 'shelf_cataloging',
    title: 'Dewey Decimal Call Number Standards',
    landmark: 'Admin Project Modal (#project-modal) > Section 05 (Physical Catalog & Librarian Accountability) / Librarian Physical Cataloging Studio > Call Number Input',
    text: `Call Number Standards for Cataloging Capstone Theses:
For Admins: Enter the Call Number in #project-call-number inside Project Modal Section 05 (Physical Catalog & Librarian Accountability).
For Librarians: Expand Library Classification & Indexing (#cn-sec-classification > #cn-call-number) in the Physical Cataloging Studio.
Format: "[Dewey Classification] / [Author Cutter/Initials] / [Year]" (Example: "025.5 / Dea / 2024").
Common CTU Program Classifications:
- 004 / 005 / 025: BSIT & BIT-Computer Software & Computing
- 621.3 / 629: BIT-Electronics & BIT-Automotive Engineering
- 647.9 / 641: BSHM Hospitality Management
- 639.2: BSFi Fisheries & Aquatic Sciences
- 370 / 371: BEEd, BSEd-Math & BTLEd Education
- 658.5: BSIE Industrial Engineering
Precaution: Section 05 fields are optional for direct Admin saves; verify Dewey classification before finalizing.`
  },
  {
    id: 'accession_number_rules',
    role_scope: ['admin', 'librarian', 'shared'],
    category: 'shelf_cataloging',
    title: 'Accession Number Uniqueness and Rules',
    landmark: 'Admin Project Modal (#project-modal) > Section 05 (Physical Catalog & Librarian Accountability) / Librarian Physical Cataloging Studio > Accession Number Input',
    text: `Accession Number Policy:
For Admins: Enter the Accession Number in #project-accession-number inside Project Modal Section 05 (Physical Catalog & Librarian Accountability), alongside #admin-catalog-audit-box.
For Librarians: Enter the Accession Number in #cn-accession inside Library Classification & Indexing (#cn-sec-classification) in the Physical Cataloging Studio.
Format: "CTU-[YYYY]-[Sequential 3+ Digits]" (Example: "CTU-2024-001").
Every physical hardbound capstone project housed in CTU Daanbantayan Library should receive a distinct institutional Accession Number so inventory tracking (#admin-catalog-audit-box) remains traceable across staff shifts.
Precaution: Never reuse a retired or previously assigned accession number across multiple bound theses.`
  },
  {
    id: 'book_photos_upload_standards',
    role_scope: ['admin', 'librarian', 'shared'],
    category: 'shelf_cataloging',
    title: 'Physical Capstone Representation Photos Standards',
    landmark: 'Project Modal > Section 04: Book & Representation Images > Drag & Drop Zone',
    text: `Standards for Uploading Book & Representation Images (Section 04):
1. Accepted Formats: PNG, JPG, and WEBP image files up to 10 MB per image, stored via Cloudinary with built-in Rotate and Mirror preview transforms.
2. In the Admin Project Modal (#project-modal), uploading Section 04 representation photos is OPTIONAL when adding or updating a project directly.
3. In Librarian Intake (#upload-metadata-modal), upload physical representation photos (front cover, signed approval sheet, system architecture diagram, and book spine).
4. Ensure text in photos is legible and avoid storing sensitive personal student ID numbers in raw photos.
Precaution: Uploaded images must remain clear and family-friendly since approved thesis visuals appear in public project detail galleries.`
  },

  // =============================================================
  // 8. LIBRARIAN PORTAL WORKFLOWS (KEPT INTACT & ISOLATED TO LIBRARIAN)
  // =============================================================
  {
    id: 'librarian_draft_submission_intake',
    role_scope: ['librarian'],
    category: 'librarian_workflow',
    title: 'Librarian Catalog Intake & Submission Workflow',
    landmark: 'Librarian Dashboard > Catalog Intake Tab > "+ Propose New Thesis"',
    text: `Librarian Staff Workflow for Cataloging Theses:
Step 1: Click "Catalog Intake" on the left sidebar rail of the Librarian Portal.
Step 2: Click the "+ Propose New Thesis" button to open the submission form (#upload-metadata-modal).
Step 3: Fill in Section 01 (Core Identity): Project Title and Authors (added individually).
Step 4: Fill in Section 02 (Academic Details): Program (BSFi, BSIT, BSHM, BSIE, BSEd-Math, BEEd, BIT-Computer, BIT-Electronics, BIT-Automotive, BTLEd-HE), Academic Year, and Faculty Adviser. Note: Status is automatically locked to "Pending Approval".
Step 5: Fill in Section 03 (Research Content) and Section 04 (Book & Representation Images): enter Abstract, research Topics, Keywords, and upload 4 to 6 physical book photos (front cover, signed approval sheet, system diagram, and spine).
Step 6: Click "Submit for Approval" (#um-submit-btn).
Status lifecycle: The proposal will enter "Pending Approval" state. It will not be visible on the public website until the Head Librarian (Admin) reviews and approves it.`
  },
  {
    id: 'librarian_submission_status_tracking',
    role_scope: ['librarian'],
    category: 'librarian_workflow',
    title: 'Checking Submission Status in Librarian Portal',
    landmark: 'Librarian Dashboard > Submissions Section > Status Badges',
    text: `Tracking Thesis Submissions in the Librarian Portal:
1. Pending Approval (Amber Badge): Awaiting Head Librarian evaluation. You may edit draft details before final approval.
2. Approved / Published (Green Badge): Accepted by Head Librarian and publicly indexed in the repository.
3. Rejected with Feedback (Red Badge): Returned by Head Librarian. Click the row to read revision comments, adjust metadata or photos, and resubmit.`
  },
  {
    id: 'librarian_permission_boundaries',
    role_scope: ['librarian'],
    category: 'librarian_workflow',
    title: 'Librarian Role Privileges & Boundaries',
    landmark: 'System Role Boundaries Notice',
    text: `Librarian Staff Permissions and Restrictions:
Permissions:
• Add new capstone thesis drafts with complete metadata and physical book photos.
• Upload book representation photos and manuscript abstracts.
• View shelf inventory and track personal submission history.
Restrictions:
• Librarians CANNOT directly publish theses to the live repository without Admin evaluation.
• Librarians CANNOT delete approved projects (deletion is locked and admin-restricted).
• Librarians CANNOT trigger Pinecone AI vector syncing or sanction/purge user accounts.`
  }
];

async function deleteDeprecatedVectors() {
  if (!DEPRECATED_IDS.length) return;
  console.log(`0. Removing deprecated vector IDs: ${DEPRECATED_IDS.join(', ')}...`);
  const deleteUrl = `https://${PINECONE_HOST}/vectors/delete`;
  const delResponse = await fetch(deleteUrl, {
    method: 'POST',
    headers: {
      'Api-Key': PINECONE_API_KEY,
      'Content-Type': 'application/json',
      'X-Pinecone-Api-Version': '2025-04'
    },
    body: JSON.stringify({ ids: DEPRECATED_IDS })
  });

  if (!delResponse.ok) {
    const errText = await delResponse.text();
    throw new Error(`Pinecone delete failed with status ${delResponse.status}: ${errText}`);
  }
  console.log('   Deprecated vectors removed (if present).\n');
}

async function generateEmbeddings(chunks) {
  const BATCH_SIZE = 20;
  const allEmbeddings = [];
  for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
    const slice = chunks.slice(i, i + BATCH_SIZE);
    const embedResponse = await fetch('https://api.pinecone.io/embed', {
      method: 'POST',
      headers: {
        'Api-Key': PINECONE_API_KEY,
        'Content-Type': 'application/json',
        'X-Pinecone-Api-Version': '2024-10'
      },
      body: JSON.stringify({
        model: 'multilingual-e5-large',
        parameters: {
          input_type: 'passage',
          truncate: 'END'
        },
        inputs: slice.map(c => ({ text: c.text }))
      })
    });

    if (!embedResponse.ok) {
      const errText = await embedResponse.text();
      throw new Error(`Embedding API failed with status ${embedResponse.status}: ${errText}`);
    }

    const embedResult = await embedResponse.json();
    allEmbeddings.push(...embedResult.data);
  }
  return allEmbeddings;
}

async function runIngestion() {
  console.log('RE-CAPS Staff Copilot Knowledge Ingestion');
  console.log(`Target Pinecone Host: ${PINECONE_HOST}`);
  console.log(`Chunks to Process: ${KNOWLEDGE_CHUNKS.length}`);
  console.log('====================================================\n');

  try {
    // Enforce strict character ceiling so multilingual-e5-large (512 tokens) never truncates content
    const oversized = KNOWLEDGE_CHUNKS.filter(c => c.text.length > 1900);
    if (oversized.length > 0) {
      throw new Error(
        `Chunks exceed 1900 chars (512-token risk): ${oversized.map(c => `${c.id} (${c.text.length})`).join(', ')}`
      );
    }

    // Step 0: Delete deprecated vectors (admin_maintenance_mode_activation)
    await deleteDeprecatedVectors();

    // Step 1: Generate embeddings via Pinecone Inference API
    console.log('1. Generating dense embeddings via Pinecone Inference API (multilingual-e5-large)...');
    const embeddings = await generateEmbeddings(KNOWLEDGE_CHUNKS);
    console.log(`Received ${embeddings.length} embeddings (${embeddings[0].values.length} dimensions each).\n`);

    // Step 2: Format vectors for upsert
    console.log('2. Preparing vector payloads with role-based metadata...');
    const vectors = KNOWLEDGE_CHUNKS.map((chunk, idx) => ({
      id: chunk.id,
      values: embeddings[idx].values,
      metadata: {
        title: chunk.title,
        category: chunk.category,
        role_scope: chunk.role_scope,
        landmark: chunk.landmark,
        text: chunk.text
      }
    }));

    // Step 3: Upsert into Pinecone
    console.log(`3. Upserting ${vectors.length} vectors into https://${PINECONE_HOST}/vectors/upsert ...`);
    const upsertUrl = `https://${PINECONE_HOST}/vectors/upsert`;
    const upsertResponse = await fetch(upsertUrl, {
      method: 'POST',
      headers: {
        'Api-Key': PINECONE_API_KEY,
        'Content-Type': 'application/json',
        'X-Pinecone-Api-Version': '2025-04'
      },
      body: JSON.stringify({ vectors })
    });

    if (!upsertResponse.ok) {
      const errText = await upsertResponse.text();
      throw new Error(`Pinecone upsert failed with status ${upsertResponse.status}: ${errText}`);
    }

    const upsertResult = await upsertResponse.json();
    console.log('Pinecone Upsert Response:', upsertResult);
    console.log(`Max chunk length: ${Math.max(...KNOWLEDGE_CHUNKS.map(c => c.text.length))} chars`);

    console.log('\n====================================================');
    console.log('INGESTION COMPLETE! All staff knowledge is live in Pinecone.');
    console.log(`Total Vectors Indexed: ${vectors.length}`);
    console.log('====================================================');
  } catch (error) {
    console.error('Ingestion Error:', error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  runIngestion();
}

module.exports = { KNOWLEDGE_CHUNKS, runIngestion };
