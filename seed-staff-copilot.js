/**
 * RE-CAPS Staff Copilot Knowledge Base Ingestion Script
 * Populates Pinecone index: recaps-staff-copilot
 * Embedding Model: multilingual-e5-large (1024 dimensions) via Pinecone Inference API
 * 
 * Supports Role-Based Metadata Scoping:
 * - role_scope: ['admin'] (Admin only: approvals, pinecone sync, delete/unlock, reports, maintenance)
 * - role_scope: ['librarian'] (Librarian only: drafting, catalog intake, submission status)
 * - role_scope: ['shared'] (Shared: shelf numbering, call numbers, accession codes, photo standards)
 */

require('dotenv').config();

const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_HOST = process.env.PINECONE_STAFF_COPILOT_HOST || 'recaps-staff-copilot-wf1edxk.svc.aped-4627-b74a.pinecone.io';

if (!PINECONE_API_KEY) {
  console.error('❌ PINECONE_API_KEY is not defined in .env');
  process.exit(1);
}

// -------------------------------------------------------------
// Knowledge Chunks Definition
// -------------------------------------------------------------
const KNOWLEDGE_CHUNKS = [
  // --- SHARED KNOWLEDGE: PHYSICAL SHELF & CATALOGING ---
  {
    id: 'shelf_location_format',
    title: 'Physical Shelf Location Formatting Rules',
    category: 'shelf_cataloging',
    role_scope: ['admin', 'librarian', 'shared'],
    landmark: 'Project Modal > Section 04: Physical Book & Shelf Details > Physical Location Note',
    text: `Physical Location Note standards for RE-CAPS library holdings:
Must specify the exact physical coordinates within the library room.
Format: Rack [Number], Row [Number], Book Stand [Number], Color [Color Label].
Example: "Rack 1, Row 2, Book Ends Stand 3, Color Blue".
Guidelines:
1. Always capitalize Rack, Row, Stand, and Color.
2. The color corresponds to the academic program binder color code (e.g. Blue for BSIT, Gold for BSHM, Maroon for BSIE, Green for BSFi).
3. This text allows library staff and students to locate the physical capstone binder in under 60 seconds.`
  },
  {
    id: 'call_number_dewey_standards',
    title: 'Dewey Decimal Call Number Standards',
    category: 'shelf_cataloging',
    role_scope: ['admin', 'librarian', 'shared'],
    landmark: 'Project Modal > Section 04: Physical Book Details > Call Number Input',
    text: `Dewey Decimal Classification Call Number guidelines:
Format: [Dewey Classification] / [Author Cutter/Prefix] / [Year].
Example: "025.5 / Dea / 2024" or "004.6 / Tan / 2023".
Rules:
1. Classification numbers for Computer Science & IT typically fall in the 004–006 range; Library & Information Sciences in 020–025; Engineering in 620s.
2. The author cutter is the first three letters of the first author's surname with initial uppercase.
3. The year is the 4-digit academic publication year.
4. Call number is printed on the physical binder spine sticker and must match the digital catalog record exactly.`
  },
  {
    id: 'accession_number_rules',
    title: 'Accession Number Uniqueness and Rules',
    category: 'shelf_cataloging',
    role_scope: ['admin', 'librarian', 'shared'],
    landmark: 'Project Modal > Section 04: Physical Book Details > Accession Number Input',
    text: `Accession Number Rules for CTU Daanbantayan Library:
Format: CTU-[Academic Year]-[Sequential 3 or 4 Digit Number].
Example: "CTU-2024-001" or "CTU-2023-042".
Important Precautions:
1. Every accession number must be globally unique across all campus holdings.
2. Duplicate accession numbers will cause catalog collision and validation rejections.
3. The accession stamp inside the physical book's title page and approval sheet must match this digital record.`
  },
  {
    id: 'book_photos_upload_standards',
    title: 'Physical Capstone Representation Photos Standards',
    category: 'shelf_cataloging',
    role_scope: ['admin', 'librarian', 'shared'],
    landmark: 'Project Modal > Section 04: Book Images > Drag & Drop Zone',
    text: `Standards for Uploading Physical Book Photos:
Requirement: Upload 4 to 6 clear photos representing the physical capstone binder.
Recommended Photo Breakdown:
1. Photo 1 (Cover): Front cover showing full gold-stamped title and CTU logo.
2. Photo 2 (Approval Sheet): Signed panel defense certificate and dean endorsement.
3. Photo 3 (Spine & Shelf): Binder spine showing call number and accession sticker.
4. Photo 4 (Architecture / System Diagram): Key hardware setup or methodology flow from the manuscript.
File Constraints: PNG, JPG, or WEBP up to 10MB per image. First uploaded image serves as default cover preview.`
  },

  // --- LIBRARIAN ROLE SCOPE: INTAKE & SUBMISSIONS ---
  {
    id: 'librarian_draft_submission_intake',
    title: 'Librarian Catalog Intake & Submission Workflow',
    category: 'librarian_workflow',
    role_scope: ['librarian'],
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
    title: 'Checking Submission Status in Librarian Portal',
    category: 'librarian_workflow',
    role_scope: ['librarian'],
    landmark: 'Librarian Dashboard > Submissions Section > Status Badges',
    text: `Tracking Thesis Submissions in the Librarian Portal:
1. Pending Approval (Amber Badge): Awaiting Head Librarian evaluation. You may edit draft details before final approval.
2. Approved / Published (Green Badge): Accepted by Head Librarian and publicly indexed in the repository.
3. Rejected with Feedback (Red Badge): Returned by Head Librarian. Click the row to read revision comments, adjust metadata or photos, and resubmit.`
  },
  {
    id: 'librarian_permission_boundaries',
    title: 'Librarian Role Privileges & Boundaries',
    category: 'librarian_workflow',
    role_scope: ['librarian'],
    landmark: 'System Role Boundaries Notice',
    text: `Librarian Staff Permissions and Restrictions:
Permissions:
• Add new capstone thesis drafts with complete metadata and physical book photos.
• Upload book representation photos and manuscript abstracts.
• View shelf inventory and track personal submission history.
Restrictions:
• Librarians CANNOT directly publish theses to the live repository without Admin evaluation.
• Librarians CANNOT delete approved projects (deletion is locked and admin-restricted).
• Librarians CANNOT trigger Pinecone AI vector syncing or toggle Maintenance Mode.`
  },

  // --- ADMIN ROLE SCOPE: PROJECTS & INTAKE ---
  {
    id: 'admin_add_new_project_procedure',
    title: 'Adding and Uploading a Capstone Project as Administrator',
    category: 'projects_cataloging',
    role_scope: ['admin'],
    landmark: 'Projects Section > Table Header Action Bar > "+ Add Project" (Emerald Green Button #add-project-btn)',
    text: `How Administrator Adds and Uploads a Capstone Project to the Repository:
Landmark: [Area: Projects Section] > [Element: "+ Add Project" Button (emerald green with + icon, #add-project-btn, located in the table action bar beside Search)].
Step 1: Click "Projects" in the left sidebar rail (#rail-nav-projects).
Step 2: Click the emerald green "+ Add Project" button (#add-project-btn) to open the Add Project modal (#project-modal).
Step 3: Fill in Section 01 (Core Identity): Enter Project Title (#project-title) and Authors (#project-authors, comma-separated full names).
Step 4: Fill in Section 02 (Academic Details): Select Academic Program from the dropdown (#project-program: BSFi, BSIT, BSHM, BSIE, BSEd-Math, BEEd, BIT-Computer, BIT-Electronics, BIT-Automotive, BTLEd-HE), Academic Year (#project-year, e.g. 2023-2024), and Faculty Adviser (#project-adviser). Status is automatically set to "Published".
Step 5: Fill in Section 03 (Research Content): Enter Abstract (#project-abstract) and search Keywords (#project-keywords).
Step 6: Fill in Section 04 (Physical Book & Shelf Details): Enter Physical Location Note (#project-location, e.g., "Rack 1, Row 2, Stand 3, Color Blue"), Call Number (#project-call-number, e.g., "025.5 / Dea / 2024"), Accession Number (#project-accession, e.g., "CTU-2024-001"), and upload 4 to 6 physical book representation photos (cover, signed certificate, spine, diagrams).
Step 7: Click "Save Capstone Project" (#save-project-btn). As an Administrator, your submission is directly published to the institutional catalog without requiring an approval review.
Precaution: Accession numbers must be unique across all institutional holdings. For bulk uploading multiple theses from a spreadsheet, click the blue "Bulk Import" button (#bulk-import-btn) instead.`
  },
  {
    id: 'admin_view_browse_existing_projects',
    title: 'Navigating, Searching, and Browsing Existing Projects in Catalog',
    category: 'projects_cataloging',
    role_scope: ['admin'],
    landmark: 'Sidebar Rail > "Projects" (2nd item from top, Document icon) > All Projects Section',
    text: `How Administrator Navigates, Searches, and Browses Existing Projects:
Landmark: [Area: Sidebar Rail] > [Element: "Projects" Nav Item (2nd tab from top, Document icon, glows emerald #08D488 when active)].
Step 1: Click "Projects" in the left sidebar rail to open the All Projects section (#section-projects).
Step 2: Table Overview: The catalog displays all published capstones with columns: Title (with Pinecone sync dot 🟢/🔴), Authors, Program, Year, Adviser, and Actions.
Step 3: Instant Filtering: Click any of the one-click Filter Pills (#project-filters) directly above the table:
• "All Projects"
• Program pills: "BSFi", "BSIT", "BSHM", "BSIE", "BSEd-Math", "BEEd", "BIT-Computer", "BIT-Electronics", "BIT-Automotive", "BTLEd-HE"
• Time filter: "Recent (30 days)"
• Sync filter: "🔴 Unsynced" (filters capstones not yet embedded into Pinecone AI search)
Step 4: Real-time Search: Type any keyword into the search bar (#projects-search) to instantly filter across title, author names, or keywords.
Step 5: Row Actions Column:
• View Details (Eye icon): Opens the project view modal with complete manuscript details and physical shelf location.
• Edit Record (Pencil icon): Opens the project edit modal (#project-modal) to update metadata.
• Unlock / Lock (Gold lock icon): Unlocks the project row to permit administrative deletion.
• Delete (Trash bin icon): Permanently purges the project from Firestore and Pinecone vector search.
Precaution: Modifying or deleting cataloged projects directly impacts the live student repository and Pinecone vector embeddings. Always unlock records with care.`
  },

  // --- ADMIN ROLE SCOPE: EVALUATIONS & APPROVALS ---
  {
    id: 'admin_approvals_evaluate_procedure',
    title: 'Evaluating and Approving Librarian Submissions',
    category: 'approvals',
    role_scope: ['admin'],
    landmark: 'Admin Sidebar Rail > Approvals (Checkmark Badge) OR Dashboard > "Pending Approvals" Card',
    text: `How Administrator Evaluates & Approves Submissions:
Landmark: Click "Approvals" in the left sidebar rail (or click the amber "Pending Approvals" KPI card on the dashboard).
Step 1: Locate the target thesis submission in the Librarian Submissions table (#approvals-table). Click the "Evaluate" button (Eye icon, #approval-review-modal trigger) in the row's Actions column.
Step 2: In the Evaluate Project Submission modal, review Section 01 (Core Information): verify project title accuracy (#approval-title) and comma-separated author names (#approval-authors).
Step 3: Review Section 02 (Academic Classification): confirm Academic Program matches CTU campus departments (BSFi, BSIT, BSHM, BSIE, BSEd-Math, BEEd, BIT-Computer, BIT-Electronics, BIT-Automotive, BTLEd-HE), Academic Year (#approval-year), Faculty Adviser (#approval-adviser), and search Keywords (#approval-keywords).
Step 4: Review Section 03 (Abstract & Summary) and Section 04 (Submitted Images): read the research abstract (#approval-abstract) and inspect the uploaded physical book representation photos in the preview grid (#approval-images-grid, default cover badge).
Step 5: Click "Approve & Publish Publicly" (Emerald button #approval-confirm-btn) to immediately publish to the live repository and queue for Pinecone AI vector search sync, or click "Reject & Discard" (#approval-reject-btn) to dismiss with reason.
Precaution: Approved projects are instantly published to the public catalog and queued for AI vector search sync. Locked projects cannot be evaluated until unlocked with the Lock button.`
  },

  // --- ADMIN ROLE SCOPE: PINECONE AI VECTOR SEARCH SYNC ---
  {
    id: 'admin_pinecone_sync_procedure',
    title: 'Syncing Projects to Pinecone AI Vector Search',
    category: 'pinecone_sync',
    role_scope: ['admin'],
    landmark: 'Projects Section > Table Header Action Bar > "Sync Pinecone" (Amber Button)',
    text: `How Administrator Syncs Theses to Pinecone AI Search:
Landmark: [Projects Section] > [Table Top Toolbar] > [Sync Pinecone (Amber Button with Lightning Icon)].
Step 1: Go to "Projects" in the sidebar rail.
Step 2: Notice the sync dots in the catalog table:
• 🟢 Green Dot: Project vector embedding is up to date in Pinecone.
• 🔴 Red Dot: Project is not yet synced; will not appear in AI semantic search or student chatbot.
Step 3: Click "Sync Pinecone" in the header toolbar.
Step 4: Confirm the sync confirmation dialog.
Step 5: Wait for the progress bar to complete all embeddings via the multilingual-e5-large pipeline.
Precaution: Embedding generation consumes API tokens and requires internet. Do not refresh or close tab during sync.`
  },

  // --- ADMIN ROLE SCOPE: EXCEL BULK IMPORT ---
  {
    id: 'admin_bulk_import_excel',
    title: 'Bulk Importing Capstone Projects from Excel',
    category: 'bulk_import',
    role_scope: ['admin'],
    landmark: 'Projects Section > Table Header Action Bar > "Bulk Import" (Blue Button)',
    text: `How Administrator Bulk Imports 40+ Theses from Spreadsheet:
Landmark: [Projects Section] > [Table Top Toolbar] > [Bulk Import (Blue Button with Upload Icon)].
Step 1: Click "Bulk Import" button beside the search bar.
Step 2: If starting fresh, click "Download Excel Template" to obtain the required column headers (Title, Authors, Year, Program, Adviser, Abstract, CallNumber, AccessionNumber, ShelfLocation).
Step 3: Drag and drop your .xlsx or .xls file into the upload zone.
Step 4: Review the Batch Validation Checklist: verify that all rows have valid 4-digit years and comma-separated author lists.
Step 5: Click "Execute Batch Import".
Precaution: Duplicate titles or accession numbers will be flagged. Existing records will not be overwritten without confirmation.`
  },

  // --- ADMIN ROLE SCOPE: UNLOCK & DELETE ---
  {
    id: 'admin_unlock_delete_project',
    title: 'Unlocking and Deleting a Cataloged Project',
    category: 'catalog_security',
    role_scope: ['admin'],
    landmark: 'Projects Table > Actions Column (Far Right) > Gold Lock Icon',
    text: `Why Delete is Disabled and How to Unlock a Project:
Explanation: By default, the red Delete button is hidden/disabled on all project rows to prevent accidental destruction of university research archives.
Procedure:
Step 1: Locate the target row in the All Projects table.
Step 2: Click the gold "Unlock" (lock-shaped) button in the row's Actions column.
Step 3: Read the safety confirmation prompt and type the required verification code.
Step 4: Once unlocked, the red "Delete" trash-bin icon becomes active and clickable.
Step 5: Click Delete and confirm permanent purge.
Precaution: Deleting a project permanently wipes it from Firestore and deletes its vector embedding from Pinecone. This action cannot be reversed.`
  },

  // --- ADMIN ROLE SCOPE: ACCREDITATION REPORTS ---
  {
    id: 'admin_print_accreditation_report',
    title: 'Generating and Printing Official Accreditation Reports for Dean / CHED',
    category: 'reports',
    role_scope: ['admin'],
    landmark: 'Analytics Section > Top Command Toolbar > "Print Official Accreditation Report"',
    text: `How Administrator Generates Institutional Research Reports:
Landmark: [Sidebar Rail > Analytics] > [Top Command Toolbar > "Print Official Accreditation Report"].
Step 1: Click "Analytics" in the left sidebar rail.
Step 2: Filter by Academic Program (e.g. BSIT, BSHM, BSFi, BSIE) and Target Academic Year.
Step 3: Review the computed intelligence graphs: Research Output by Program, Annual Intake Timeline, and Faculty Advisers breakdown.
Step 4: Click "Print Official Accreditation Report" in the top-right toolbar.
Step 5: An official, formatted printable report with CTU Daanbantayan Campus header, date seal, and verified thesis counts will open for PDF export or paper printing.
Precaution: Only approved, cataloged manuscripts are counted toward institutional accreditation metrics.`
  },

  // --- ADMIN ROLE SCOPE: CREATE LIBRARIAN STAFF ---
  {
    id: 'admin_create_librarian_staff',
    title: 'Creating a Librarian Staff Account',
    category: 'user_management',
    role_scope: ['admin'],
    landmark: 'Users Section > Header Toolbar > "+ Create Librarian" (Emerald Button)',
    text: `How Administrator Registers a New Library Staff Member:
Landmark: [Sidebar Rail > Users] > [Top Header Toolbar > "+ Create Librarian" (Emerald Button)].
Step 1: Navigate to "Users" in the sidebar rail.
Step 2: Click "+ Create Librarian".
Step 3: Enter the staff member's Full Name, Institutional CTU Email (@ctu.edu.ph), and an initial temporary password.
Step 4: Click "Register Librarian".
Step 5: Copy the generated credentials card (Name, Email, Password) to hand off to the staff member.
Precaution: The staff member will be granted librarian cataloging privileges and can immediately log into the Librarian Portal.`
  },

  // --- ADMIN ROLE SCOPE: MAINTENANCE MODE ---
  {
    id: 'admin_maintenance_mode_activation',
    title: 'Activating Maintenance Mode for Upgrades and Outages',
    category: 'system_settings',
    role_scope: ['admin'],
    landmark: 'Settings Section > General Settings Card > "Maintenance Mode" Switch',
    text: `How Administrator Enables Maintenance Mode:
Landmark: [Area: Settings Section] > [Element: "Maintenance Mode" Switch (General Settings Card)].
Step 1: Click "Settings" in the left sidebar rail.
Step 2: In the General Settings box, find the "Maintenance Mode" switch.
Step 3: Toggle the switch to ON.
Step 4: Confirm activation in the dialog prompt.
Effect: The public search portal and student login will be immediately locked and redirected to maintenance.html (showing scheduled completion date and trivia game). Logged-in Administrators retain full dashboard access to perform data work.
To restore public access: Toggle the switch back to OFF.`
  },

  // --- ADMIN ROLE SCOPE: PROJECT EDIT & UPDATE ---
  {
    id: 'admin_edit_existing_project',
    title: 'Editing and Updating an Existing Cataloged Thesis',
    category: 'projects_cataloging',
    role_scope: ['admin'],
    landmark: 'Projects Table > Target Thesis Row > "Edit Record" (Pencil Icon #edit-project-btn)',
    text: `How Administrator Edits an Existing Project in the Catalog:
Landmark: [Area: Projects Section > All Projects Table] > [Element: "Edit Record" (Pencil Icon in Actions Column)].
Step 1: Navigate to "Projects" in the left sidebar rail.
Step 2: Locate the thesis in the table using the search box (#projects-search) or Program Filter Pills.
Step 3: Click the gold Pencil icon (Edit) in the row's Actions column.
Step 4: In the Edit Capstone Project modal, update title, author names, academic program, abstract, keywords, or physical shelf location note (Rack, Row, Stand, Color).
Step 5: To replace or add physical book photos, drag new photos into Section 04 (Representation Photos).
Step 6: Click "Update Capstone Project" (#save-project-btn). Changes will immediately update Firestore and queue Pinecone AI embeddings for automatic re-indexing.
Precaution: Modifying title or abstract alters the vector embedding representation; verify data integrity before updating.`
  },

  // --- ADMIN ROLE SCOPE: USER ACCOUNT GOVERNANCE ---
  {
    id: 'admin_user_account_management',
    title: 'Managing User Accounts, Unlocking Logins, and Password Resets',
    category: 'user_management',
    role_scope: ['admin'],
    landmark: 'Users Section > Users Table > Actions Column',
    text: `How Administrator Manages Student, Faculty, and Staff Accounts:
Landmark: [Area: Users Section > Users Table] > [Element: Actions Column (Key Icon & Unlock Icon)].
Step 1: Click "Users" in the left sidebar rail.
Step 2: Use the Role Filter Pills (All Users, Students, Faculty, Librarians, Administrators) or search box (#users-search) to find the user by name or CTU email.
Step 3: To reset a password: Click the Key icon in the user row. A temporary password modal will generate verified credentials.
Step 4: To unlock a locked account: If a user exceeded failed login attempts, click the gold Unlock/Shield button to restore access immediately.
Step 5: To adjust account privileges: Click the Edit icon to transition a user between Student, Faculty, or Librarian roles.
Precaution: Administrator privileges cannot be downgraded from this table to prevent institutional lockout.`
  },
  {
    id: 'admin_delete_user_account',
    title: 'Deleting a User Account, Deletion Safety, and Lockout Protection',
    category: 'user_management',
    role_scope: ['admin'],
    landmark: 'Users Section > Users Table > Actions Column (Red Trash Can Icon)',
    text: `How Administrator Deletes a User Account:
Landmark: [Area: Users Section > Users Table] > [Element: Actions Column (Red Trash Can Icon)].
Step 1: Click "Users" in the left sidebar rail to open the Users management section (#section-users).
Step 2: Use the Role Filter Pills (All Users, Students, Faculty, Librarians) or type the name or CTU email in the search box (#users-search) to locate the target user.
Step 3: Hover over the user's row and locate the Actions column on the right.
Step 4: Click the red Trash Can icon to trigger the account deletion modal.
Step 5: Review the safety confirmation dialog: verify the user's email and confirm that they have no pending project proposals or evaluations.
Step 6: Click "Delete" to permanently remove the account.
Precaution: Deleted accounts are permanently purged from Firebase Authentication and Firestore records. User accounts cannot be restored once deleted. Administrator accounts cannot be deleted to prevent institutional lockout.`
  },

  // --- ADMIN ROLE SCOPE: AUDIT LOGS & TRACEABILITY ---
  {
    id: 'admin_system_audit_logs_analytics',
    title: 'Viewing System Activity and Compliance Audit Logs',
    category: 'system_audit',
    role_scope: ['admin'],
    landmark: 'Sidebar Rail > "Activity" (Clock Icon) OR Analytics Section',
    text: `How Administrator Inspects Repository Audit Logs:
Landmark: [Area: Sidebar Rail] > [Element: "Activity" (Clock Icon, #rail-nav-activity)].
Step 1: Click "Activity" in the left sidebar rail to view the system audit trail.
Step 2: Inspect chronological logs of all institutional actions: project uploads, evaluations, deletions, user logins, and Pinecone vector syncs.
Step 3: Filter logs by action type or date range to audit librarian cataloging contributions or security events.
Step 4: Click any entry to inspect actor email, timestamp, IP address, and changed fields.
Precaution: System audit trails are immutable and permanent for CHED accreditation compliance.`
  },

  // --- ADMIN ROLE SCOPE: TROUBLESHOOTING AI SEARCH ---
  {
    id: 'admin_troubleshoot_rag_catalog',
    title: 'Troubleshooting Why a Thesis is Missing from Search or RAG',
    category: 'pinecone_sync',
    role_scope: ['admin'],
    landmark: 'Projects Section > Filter Pills > "🔴 Unsynced" Pill (#filter-unsynced)',
    text: `How Administrator Troubleshoots Missing Theses in Search or Student AI Chat:
Landmark: [Area: Projects Section > Filter Pills] > [Element: "🔴 Unsynced" Filter Pill (#filter-unsynced)].
Step 1: Navigate to "Projects" in the left sidebar rail.
Step 2: Check for Unsynced Status: Click the "🔴 Unsynced" filter pill above the table. If the project appears here, its vector embedding has not been indexed in Pinecone.
Step 3: Check Approval Status: Go to "Approvals" in the sidebar rail. If the thesis was drafted by a librarian, it must be evaluated and approved before it enters the public index.
Step 4: Trigger Pinecone Sync: In Projects, click the amber "Sync Pinecone" button (#sync-pinecone-btn) in the header action bar.
Step 5: Verify Green Dot: Once sync finishes, verify that the green dot ( 🟢 ) appears beside the thesis title. The thesis is now fully searchable in both keyword and AI RAG modes.
Precaution: The student AI chatbot and semantic RAG search only retrieve theses that have green synced status and "Published" catalog state.`
  }
];

// -------------------------------------------------------------
// Ingestion Pipeline Function
// -------------------------------------------------------------
async function runIngestion() {
  console.log('====================================================');
  console.log('🚀 RE-CAPS Staff Copilot Knowledge Ingestion');
  console.log(`📌 Target Pinecone Host: ${PINECONE_HOST}`);
  console.log(`📚 Chunks to Process: ${KNOWLEDGE_CHUNKS.length}`);
  console.log('====================================================\n');

  try {
    // Step 1: Generate embeddings via Pinecone Inference API
    console.log('1️⃣ Generating dense embeddings via Pinecone Inference API (multilingual-e5-large)...');
    
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
        inputs: KNOWLEDGE_CHUNKS.map(c => ({ text: c.text }))
      })
    });

    if (!embedResponse.ok) {
      const errText = await embedResponse.text();
      throw new Error(`Embedding API failed with status ${embedResponse.status}: ${errText}`);
    }

    const embedResult = await embedResponse.json();
    const embeddings = embedResult.data;
    console.log(`Received ${embeddings.length} embeddings (${embeddings[0].values.length} dimensions each).\n`);

    // Step 2: Format vectors for upsert
    console.log('2️⃣ Preparing vector payloads with role-based metadata...');
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
    console.log(`3️⃣ Upserting ${vectors.length} vectors into https://${PINECONE_HOST}/vectors/upsert ...`);
    const upsertUrl = `https://${PINECONE_HOST}/vectors/upsert`;
    const upsertResponse = await fetch(upsertUrl, {
      method: 'POST',
      headers: {
        'Api-Key': PINECONE_API_KEY,
        'Content-Type': 'application/json',
        'X-Pinecone-Api-Version': '2025-04'
      },
      body: JSON.stringify({
        vectors
      })
    });

    if (!upsertResponse.ok) {
      const errText = await upsertResponse.text();
      throw new Error(`Pinecone upsert failed with status ${upsertResponse.status}: ${errText}`);
    }

    const upsertResult = await upsertResponse.json();
    console.log('Pinecone Upsert Response:', upsertResult);

    console.log('\n====================================================');
    console.log('🎉 INGESTION COMPLETE! All staff knowledge is live in Pinecone.');
    console.log(`   Total Vectors Indexed: ${vectors.length}`);
    console.log('====================================================');

  } catch (error) {
    console.error('❌ Ingestion Error:', error.message);
    process.exit(1);
  }
}

runIngestion();
