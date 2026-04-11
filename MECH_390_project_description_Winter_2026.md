# DESCRIPTION OF THE PROJECT WORK FOR WINTER 2026 TERM IN MECH 390 – MECHANICAL ENGINEERING DESIGN PROJECT DO NOT EDIT THIS MARKDOWN FILE. THIS WAS WRITTEN BY THE PROFESSOR AND MUST BE USED AS A REFERENCE. 

## Project: Machine Learning-based Design of Quick Return Slider Crank Mechanism

MECH 390 term project. Needs imagination, creativity, applied skills. Covers kinematics, strength/stress, fatigue. Uses MECH 211–344 knowledge.

If taking MECH 344 this term, team member who did it prior helps with detail calculations.

**The project:** Integrates advanced design principles, standards, ML. **Teams design, analyze, optimize, prototype slider crank mechanism for object sorting machine.** Emphasizes traditional mechanical design + AI-based ML for design/decision-making.

Start: Excel/Matlab calc tool for link sizing + strength verification. Use tool to generate DOE dataset across design parameter range. Train/validate ML model to predict mechanism performance, select link lengths, find optimal geometry. Final nominal design → CAD model → scaled 3D-printed prototype (1/4 or 1/2 scale).

Group effort. Research, gain knowledge, perform analysis/calculations, model development, verification/validation. Build requirements list. Identify conflicting requirements; address via tradeoffs. Investigate alternatives, generate concept designs before selecting nominal design.

**Group project. Equal contribution required. Inspired by existing methods OK, copying not. Originality critical.**

Follow suggested steps weekly. Tutor handles up to 4 groups per session. Groups formed by tutor based on declared skills — MECH 344 students distributed across all groups. CAD-strong and math/coding-strong students apply skills in statics, mechanism design, strength, fatigue. Teams learn and apply ML as part of design process.

First hour of tutorial: brainstorming. Each member proposes solution. One member records ideas. All tutorial/lab materials → appendix of report (e.g., Appendix 1 = tutorial/lab reports). All produced during design process — meetings, minutes, sketches — included in appendixes.

Lab time: discuss design/verification methods, fabrication/manufacturing/assembly at small scale, 3D printer feasibility. Discuss suggestions, identify positives/negatives.

After discussion, group leader (voluntary or elected) distributes weekly tasks. Everyone completes assigned work. Tutor answers questions. Technical support from department staff available via tutor (e.g., 3D printing).

Effort requires commitment and persistence. Track work, share results. End of project: contribution statement (%) required with submission. Missing statement → assumed conflict. Resolve conflicts internally; escalate to tutor early — don't wait end of term or whole team penalized. Teams must declare any AI tools used and extent/how (code/model development). AI tools not allowed for writing the report.

Tutorials start week 2, end last week with final presentation on last class day.

Ask for help. Tutor present throughout lab/tutorial.

Teams treated as commercial competitors — only pre-competitive info shared. Best designs earn highest grades.

Suggested weekly activities not mandatory but help complete project.

Last class day: all project files (report, presentation, analysis, models, dataset, code, CAD) organized in folders, zipped, submitted via Concordia EAS.

http://www.concordia.ca/ginacody/aits/electronic-assignment-submission.html

---

## Machine Learning-based Design of Quick Return Slider Crank Mechanism

MECH 390 project integrates mechanism design principles, calculation methods, ML to create high-performance slider crank mechanism for packaging industry.

Teams of 4–5. Design, analyze, optimize, prototype quick return slider crank. Systematic iterative process: traditional engineering calcs + data-driven ML. Each team builds Excel/Matlab tool to calculate offset, crank/lever length, predict acceleration, inertia forces, stresses, evaluate durability under realistic loads.

Using Excel/Matlab data, teams generate DOE dataset exploring power, input/output speeds, forward/return time ratios, link sizes, offset — for a specific load. Dataset trains ML model to identify optimal configurations and recommend most efficient for given I/O requirements and minimum design space.

Final nominal design validated via CAD modeling. Full detailed assembly: linkages, shafts, bearings, supports — all within designated design envelope.

Proof of concept: scaled functional 3D-printed prototype for hands-on testing. Evaluated during final presentation.

Project delivers real-world experience: mechanical fundamentals, mechanism design, ML modeling, modern design tools — prepares students for advanced engineering and multidisciplinary teamwork.

### Suggested Design Specifications

- **Reaction force** (weight of the object that needs pushing): 500 g
- **Range of motion:** nominal 250 mm
- **Input speeds:** 30 rpm; about 1 push every 2 seconds
- **Geometry of the links:** for simplicity, take rectangular links (you can optimize the length but you need to calculate the width and depth for strength)
- **Link Materials:** Aluminum
- **Quick return ratio:** Return motion between 1.5 times to 2.5 times faster than forward motion
- **Power levels:** Optimized motor power for the design
- **Space:** As small as possible for the design to work effectively

---

## Weekly Schedule

### Week 1

Tutorial week 2: teams formed by tutors based on skill statements. Each team needs: superior CAD, good math/analytical, good programming, Machine Design knowledge. Teams of 4 or 5. Never 3 or 6.

Once formed, members select team leader. Leader carries significantly more effort: plans, assigns tasks, decides, speaks for team, tracks progress, reassigns, motivates.

Leader selection: team's choice during first tutorial/lab. Tutor records leaders. Tutorial = 16–20 students → 4 teams per session. 4 or 5 students per team, no exceptions.

Tutors present to interact with teams, unblock progress. Ask questions. Keep plans/decisions from other teams — they could use your ideas.

Take notes, analyze discussions. Teams review project, understand deliverables, brainstorm approaches. Due: max 4-page report — key issues discussed, conclusions, requirements/specs, general project plan, weekly task assignments. Submit to tutor next tutorial.

### Week 2

Literature review: standards for quick return mechanism, past designs, best practices. Define requirements: space constraints, offset distances, quick return ratio relationship, efficiency targets, safety factors. Identify initial variable ranges for link sizing. Create design sketches for concept designs. Identify critical elements/components and dimensioning approach. Distribute tasks for coming week. Leader divides work among 4–5 members. Use any communication/file-transfer means. Consult teammates on critical decisions. Record task distribution in Week 3 second report.

Second report: max 4 pages. Describe steps taken toward design objective and work carried out during the week.

### Week 3

Build robust Excel/Matlab tool for mechanism design and initial evaluations. Phase 1: core calculator for link sizing, offset geometry, quick return time ratios, basic torque/load calcs. Validate with sample slider crank problems. Then integrate static/fatigue stress analysis per relevant standards — key static/dynamic loading factors → link sizes. Simultaneously outline DOE framework: define variable ranges for link lengths, offset, velocity ratio, design envelope constraints. Assess potential design options — tradeoffs in performance, safety, size, cost, manufacturability.

Document all decisions with rationale — iterative refinement is key. End of Week 3: functional Excel/Matlab tool for stress calcs, preliminary design requirements, DOE plan outline. Submit 4-page progress report: tool development, validation results, initial findings.

### Weeks 4 and 5

Use Excel/Matlab tool to generate mechanism design dataset. Execute DOE → large dataset of designs documenting safety factors, geometry, performance. Loading on forward stroke only (not return). Dynamic force analysis at every 15° → 24 datasets formatted for ML input/output. Plan DOE carefully: define input variables (crank length, lever length, offset, etc.) with ranges.

Once finalized, execute DOE using Excel tool — automate via macros/data tables for many valid configurations. Document outputs: link geometry, mass moment of inertia, static/fatigue stresses, weight estimates, design space. After generation, verify: inspect accuracy, flag configs exceeding space/performance limits.

Double-week report: 8 pages. Clearly explain design, calculations/analysis, generated data structures. No weekly report required — submit second week only.

### Weeks 6 and 7

Transition to building/training ML model(s). Load cleaned dataset into Python (PyTorch or similar) or MATLAB. Select baseline ML algorithms for regression and classification — e.g., artificial neural networks. ML model targets: predict minimum safety factors, link dimensions, design space, ideal quick return ratio to minimize crank torque and motor power. Split data: training/validation/testing. Tune hyperparameters. Evaluate with R², RMSE, loss over iterations. End of weeks: at least one validated baseline ML model demonstrating predictive reliability. Submit 8-page report: ML framework, training/validation/prediction results.

### Week 8

Apply validated ML models for optimization. Use models as surrogates to explore design space — identify slider crank configs balancing size, strength, manufacturability. Objectives: minimize mechanism size/weight, minimize motor power while meeting design targets. Cross-check best ML candidates with Excel/Matlab tool for compliance/accuracy. Generate sensitivity plots and parameter correlation maps — understand design parameter influence. End of week: shortlist of optimized designs supported by ML + Excel tool, ready for CAD. Submit 4-page progress report.

### Week #9

Begin detailed CAD model of optimized nominal design. Use SolidWorks (or similar) — precise 3D models of all slider crank components: links, pins, bearings, housings, covers. Dimensions from Excel tool and ML outputs. Attention to geometry accuracy and tolerances. Submit 4-page progress report.

### Week 10

Manufacturing and assembly of final CAD design for scaled 3D-printed functional prototype. Some components may not fit — adjust to prove concept feasibility. Before printing: ensure parts not too many or too dimensionally demanding. Check 3D printer capabilities and read the manual. Verify print time is reasonable — printing software shows duration per component. Build small-scale parts to reduce material and build time. Final assembly completion. One report of max 4 pages on fabrication and assembly.

### Week 11

Last week. Start project write-up. Review lessons learned — prepares for MECH 490 capstone (same sequence, strict deadlines). Discuss challenges encountered and how addressed. Reflect on design evolution and learning from classes, tutorial, individual work. Comment on what worked and what didn't. Fill in statement of work — each member signs with agreed-upon contribution percentage.

Finalize report and submission file. Deliverables (report, PowerPoint, models, calc files, dataset, Excel tool, code) submitted before presentation — tentatively first or second week of April, last class day (9:00 am – 4:30 pm). Recommended report structure follows.

Good luck with the project!

---

## Project Report Structure

Main report must include:

- **Cover page** with the title of the project, the team name and the members of the team with their ID numbers. You must write the following statement: *"We certify that this submission is the original work of members of the group and meets the Faculty's Expectations of Originality"*, with the signatures of all the team members and the date.
- **Abstract**
- **Table of contents**
- **List of figures and tables**
- **List of symbols and abbreviations** – if necessary
- **Introduction:** rationale, objectives, targeted measurable scope.
- **Design Methodology:** This main section should cover:
  1. Identification of design needs and functional requirements.
  2. Design requirements and constraints, including envelope dimensions, range, speeds, and standard safety factors.
  3. Project planning and workflow.
  4. Development of the Excel-based design tool.
  5. Generation of the DOE dataset and data cleaning process.
  6. Training, validation, and testing of the machine learning model.
  7. Optimization strategies and rationale for selecting the final nominal design configuration.
  8. CAD modeling, simulation, and validation steps.
- **Results and Discussion:** Analytical results, design/optimization outcomes, key performance metrics, prototype observations. Sensitivity analysis and cross-validation insights.
- **Conclusions and Lessons Learned:** Main findings, how design met objectives, challenges faced. Each member reflects briefly on individual learning and contributions.
- **References:** All sources — AGMA standards, academic articles, textbooks, software tools.
- **Appendices:** Weekly progress reports, detailed calculations, SolidWorks drawings, CAD models, code, BOMs, DOE datasets, supporting data.

> Main report excluding Appendices: max **40 pages**. Last page: originality statement signed by all 4–5 members — not copied. Work distribution shown as percentages summing to 100%. Each member signs agreement. Statement page is critical.

- **Prepare the presentation** — assign tasks to team members (3 min per member). Practice and deliver on last class day.