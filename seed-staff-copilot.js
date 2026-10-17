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
Step 2: Click the green "+ Propose New Thesis" button.
Step 3: Enter Title, Authors (comma-separated), Academic Program, Adviser, Batch Year, and Abstract.
Step 4: Fill in Section 04: Physical Shelf Location Note, Call Number, and Accession Number.
Step 5: Upload 4 to 6 representation photos.
Step 6: Click "Submit for Admin Approval".
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
• Add new capstone thesis drafts with physical shelf metadata and call numbers.
• Upload book representation photos and manuscript abstracts.
• View shelf inventory and track personal submission history.
Restrictions:
• Librarians CANNOT directly publish theses to the live repository without Admin evaluation.
• Librarians CANNOT delete approved projects (deletion is locked and admin-restricted).
• Librarians CANNOT trigger Pinecone AI vector syncing or toggle Maintenance Mode.`
  },

  // --- ADMIN ROLE SCOPE: EVALUATIONS & APPROVALS ---
  {
    id: 'admin_approvals_evaluate_procedure',
    title: 'Evaluating and Approving Librarian Submissions',
    category: 'approvals',
    role_scope: ['admin'],
    landmark: 'Admin Sidebar Rail > Approvals (Checkmark Badge) OR Dashboard > "Pending Approvals" Card',
    text: `How Administrator Evaluates & Approves Submissions:
Landmark: Click "Approvals" in the left sidebar rail (or click the amber "Pending Approvals" KPI card).
Step 1: In the Pending Approvals table, find the thesis row and click the green "Evaluate" button.
Step 2: Review core metadata: verify thesis title, correct author capitalization, and designated adviser.
Step 3: Review Section 02 Classification: ensure academic program matches the curriculum.
Step 4: Review Section 04 Physical Shelf: confirm Shelf Location Note, Call Number, and unique Accession Number.
Step 5: Click "Approve & Publish" (Emerald button) to immediately release to the public repository, or click "Reject" to return with remarks.
Precaution: Approved projects are instantly published and queued for AI vector search sync.`
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
Landmark: [Sidebar Rail > Settings] > [General Settings Card > "Maintenance Mode" Toggle].
Step 1: Click "Settings" in the left sidebar rail.
Step 2: In the General Settings box, find the "Maintenance Mode" switch.
Step 3: Toggle the switch to ON.
Step 4: Confirm activation in the dialog prompt.
Effect: The public search portal and student login will be immediately locked and redirected to maintenance.html (showing scheduled completion date and trivia game). Logged-in Administrators retain full dashboard access to perform data work.
To restore public access: Toggle the switch back to OFF.`
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
    console.log(`✓ Received ${embeddings.length} embeddings (${embeddings[0].values.length} dimensions each).\n`);

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
    console.log('✓ Pinecone Upsert Response:', upsertResult);

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
