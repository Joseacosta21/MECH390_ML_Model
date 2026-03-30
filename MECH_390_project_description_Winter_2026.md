# DESCRIPTION OF THE PROJECT WORK FOR WINTER 2026 TERM IN MECH 390 – MECHANICAL ENGINEERING DESIGN PROJECT

## Project: Machine Learning-based Design of Quick Return Slider Crank Mechanism

This term project in MECH 390 requires imagination, creativity and practical application of acquired new knowledge and skills. It will be required to carry out calculation of performance, kinematics, strength/stress and fatigue, among other types of analyses. You will be also using knowledge that you already have from MECH 211, MECH 213, MECH 243, MECH 244., …. to MECH 343 and MECH 344.

If you take Machine Element Design - MECH 344 this term, hopefully someone in your team did it already and he/she will be able to help with the detail calculations as the sequence in the project may be different than the one in MECH 344.

**The project:** This term's project integrates advanced design principles, standards, and machine learning techniques. **Teams will design, analyze, optimize, and prototype a slider crank mechanism for an object sorting machine.** The project will emphasize both traditional mechanical design and machine learning on the basis of artificial intelligence (AI) for design and decision-making.

The process will begin with developing an Excel or Matlab based calculation tool for link sizing and strength verification. This tool will be used to generate a design of experiments (DOE) dataset across a range of design parameters. Teams will then train and validate a machine learning model to predict the performance of the mechanism, select link lengths, and identify optimal design geometry. The final nominal design will be modeled in CAD and demonstrated via a scaled 3D-printed functional prototype (1/4 scale or 1/2 scale).

This project is a group effort. You should discuss, do research and gain new knowledge, perform analysis/calculations, model development, verifications/validations, make decisions and provide evidence that your design method(s) are the suitable one(s). Build a detail list of requirements. Establish which of the requirements are conflicting and how you could address the conflicting requirements by performing tradeoff among them. Teams shall investigate the potential/alternative solutions and generate various concept designs with sufficient dataset before a proposed nominal design can be selected and finalized.

**This is a group project, and everybody must contribute equally to the realization of the final output. Teams can be inspired from existing computational methods/tools to some extent, but not copy them. Originality of your design and computational methods are very important.**

It is suggested you few steps to follow throughout the term. You will encounter every week the tutor who is in charge with up to four projects (maximum four groups in the tutorial/laboratory session). Each design group will be formed by students selected by the tutor based on their declared skills. For an example, the students who have already taken MECH 344 will be distributed in all groups as their knowledge and view is important to the team. Same for the students who are very comfortable with CAD design or with mathematical calculations and computer software/coding can use their skills in static, mechanism design, strength and fatigue analyses of proposed design(s). Teams will study, learn and apply a proper machine learning method as a part of their design process.

For the first hour of the tutorial time – students can have a brainstorming session. Each member of the group will have his or her time to propose a solution based on their past experience and knowledge. One of the team members should write down these ideas. Please note that all materials resulted from the tutorial/lab session are part of the design project and must be included in the appendix of the project report. For an example, the appendix 1 will include tutorial/laboratory reports. Hence, all that is produced during the design process – meetings, minutes, sketches, report – they are part of the project report in the appendixes.

For the lab time, teams should discuss design and verification methods of designed parts for functionality. Teams should study fabrication, manufacturing and assembly of possible parts/components in small scales topics, feasibility of parts/components to be fabricated using 3D printers. However, teams can also discuss in group the suggestions and identify the positive as well as the negative points of the stated ideas.

Following this discussion that will reveal a design path direction, the group leader – voluntary or elected (each team is free to do the selection the way they find suitable) will distribute the tasks for the week to each team member. Everyone has to come back with the "homework done". The process could be influenced by the tutor which has the task to answer your questions. Technical support from department technical staff can be sought with the coordination of the tutor (such as printing parts using 3D printers).

The required effort needs commitment and persistence from all team members. The team members need to keep track of the carried-out work and exchange their results with the other team members. At the end of the project, a statement which includes the contribution of all individuals in the team measured in percentage must be transmitted along with the project. If such statement is not included, it is automatically assumed that there is a conflict in that team. Team members are expected to resolve any conflicts inside the formed team. If not, they need to raise the issue with their tutor, don't wait the end of term, otherwise a whole team may be penalized on the basis of poor team work. It is also required that teams must declare in their report if any AI tool(s) and what extend (detailed description how and what part of their project for example code/model development) are used in any part of their project. However, it is not allowed to use any AI Tool in writing the report.

The first tutorial/lab session will start the 2nd week of the term and ends last week when the teams will present the results of their work during the last class day.

Do not forget: if you need help, ask. The tutor is in the class at all times during the lab/tutorial period.

We shall assume that the teams are in commercial competition such that only pre-competitive information is exchanged among the teams. The best designs will secure high grades.

The suggested activities described for weeks here are not mandatory, but the suggestions made will help you to complete the design project.

On last class day, all project files (report, presentation, analysis results, models, generated dataset, code scripts, CAD files) to be organized under different folders and zipped and submitted through Concordia Electronic Assignment Submission (EAS). Teams will submit project deliverables using Concordia Electronic Assignment Submission (EAS) in the link below.

http://www.concordia.ca/ginacody/aits/electronic-assignment-submission.html

---

## Machine Learning-based Design of Quick Return Slider Crank Mechanism

This term's MECH 390 design project integrates advanced mechanism design principles, calculation methods and machine learning-based technique(s) to create a practical, high-performance slider crank mechanism for pushing objects for packaging industry machine applications.

Students will work collaboratively in teams of four to five to design, analyze, optimize, and prototype a quick return slider crank mechanism. The project emphasizes a systematic, iterative design process that utilizes traditional engineering calculations with data-driven machine learning methods. Beginning with fundamental analytical design based on standards, each team will develop a customized Excel or Matlab based tool to calculate offset, crank length and lever length, predict acceleration, inertia forces and stresses, and evaluate durability under realistic load conditions.

Building on this analytical foundation using the Excel spreadsheet or Matlab data, teams will generate a Design of Experiments (DOE) dataset to explore a broad range of power, input and output speeds, ratios of time difference between forward and return motion, sizes of the links and offset, for a specific load. This dataset will then be used to train a machine learning (ML) model capable of identifying optimal configurations and design space and recommending the most efficient for the given input-output requirements and minimum design space.

The final nominal design will be validated against using CAD modeling, teams will prepare a fully detailed assembly, including linkages, shafts, bearings, supports, ensuring that all components fit within the designated design envelope.

To demonstrate proof of concept for the nominal design, each team will produce a scaled, functional prototype using 3D printing technologies by allowing hands-on testing of the mechanisms functionality/motion. The prototype will serve as a tangible representation of the design and will be evaluated during the final presentation.

This project provides comprehensive, real-world experience in mechanical system design, integrating mechanical engineering fundamentals, mechanism design, machine learning modeling, and modern design tools by preparing students for advanced engineering challenges and multidisciplinary teamwork in professional practice.

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

During the first tutorial week (second week of the term), teams will be formed. The tutors will establish the teams based on individual statements related to the skills of each team member. In principle, each team will include students with the following skills: superior CAD skills, good math and analytical skills, good programming skills, knowledge of Machine Design. The teams are made of 4 (four) or if not possible due to the number of students in the group, 5 (five). Never 3 or 6.

Once the teams are formed, the members will select their team leader. Be aware that the team leader is going to put significantly more effort than the other team members as s/he plans, assigns tasks and takes decisions for the team, speaks for the team, is in charge with the advancement of the project, s/he changes assignments among the team members, motivates the team, etc.

Selection of the team leader will be made by the team members – the groups will have the first tutorial/lab for this task as above mentioned. At the end of the two hours, each team will have a leader, and the tutor will take note of the team leaders. The tutorial/laboratory are usually made of 16–20 students – this means four teams for each tutorial session. It is required that a team will include four or five students without any exceptions.

Lab instructors/tutors are in the lab/tutorial to interact with teams and help the smooth progress of work (in case a team gets "stacked"). Ask questions, ask opinion of the team members. As this is a group project, it is supposed to keep the progress of work, the plans and the decisions away from the "competition" which is represented by the other teams. They could get "inspired" from your idea and come up with a better project than yours. When one has a reference, it is easier to figure out which of the two ideas might be better.

One should take notes during the tutorial/lab time, have an analysis of the discussions. Teams will review the project, understand deliverables, and brainstorm design ideas and approaches for the slider crank mechanism. The team is due a max. 4 pages report which will present the essential issues that the team has discussed, the conclusions of the discussions, the few established specifications/requirements of the design and the plan for the project development in general. The report will include the assignments/tasks of each team member done for that week. The report is due next week during the tutorial to the tutor.

### Week 2

Literature review analysis and understanding of the problems associated with the design project should be carried out. Teams will conduct a detailed review of standards for quick return mechanism, design analysis, examine past mechanism designs, and summarize best practices. Requirements such as space constraints, offset distances and the relationship to ratio of quick return, efficiency targets, and safety factors will be defined. Initial variable ranges for link sizing will be identified. Design sketches can be created for different concept designs. Identify the elements/components that appear as critical and decide upon the way they would be dimensioned. At this time, the tasks should be distributed for the coming week. The group leader should divide the assignment in the number of team members (4/5) and proceed with the work. The team may use any means they consider suitable to communicate and transfer files. The work during the week should be carried out consulting the teammates for the important decisions (the critical decisions that will shape the output of the work). The distribution of tasks should be recorded and included in the third week, second report.

The second report should not exceed 4 pages, and it would explain the best way possible the taken steps towards attaining the design objective. The report should be a description of the work carried out during the week.

### Week 3

Teams will focus on developing a robust Excel or Matlab tool for the mechanism design and performing initial design evaluations. The first phase will include building the core calculator for link sizing and offset geometry, quick return time ratios, and basic torque and load calculations, followed by validation with sample slider crank mechanism problems to ensure accuracy. Once the basic tool is validated, teams will integrate static and fatigue stress analysis per relevant standards, incorporating key factors for static and dynamic loading to come out with the size of the links. In parallel, teams will begin outlining Design of Experiments (DOE) framework by defining variable ranges for link lengths, offset lengths, velocity ratio between forward and return, and design envelope constraints. Initial assessments of potential design options will be performed to understand trade-offs in performance, safety, size, cost, and manufacturability.

Teams should also document all decisions, including the rationale for chosen parameters, since iterative refinement will be a key part of the process. By the end of Week 3, teams should have a functional Excel or Matlab tool capable of calculating stresses, a preliminary set of design requirements, and an outline for the Design of Experiments (DOE) plan. A four-page progress report summarizing the tool development, validation results, and initial findings must be submitted.

### Weeks 4 and 5

Teams will use the Excel or Matlab tool to generate the dataset for the mechanism design. Teams will execute the DOE to produce a sufficiently large dataset of designs, documenting safety factors, geometry, and performance metrics. Since the loading on the mechanism will be only on the forward, and not on the return, dynamic force analysis to compute the torque requirement together with the inertial forces, at every 15° will provide 24 datasets that will be formatted and structured for machine learning input and output. Teams will start by carefully planning the DOE. This involves defining all input variables such as, the length of the crank, length of the lever, offset distance, etc., along with their corresponding ranges.

Once the variables and ranges are finalized, the teams will execute the DOE using the Excel tool. This step requires automation through macros or data tables to efficiently generate many valid configurations. For each configuration, critical design outputs will be documented, including link geometry, mass moment of inertia as well as key performance metrics like static and fatigue stresses. Weight estimates and design space should also be captured for later packaging and performance analyses. After the dataset is generated, teams will perform thorough verification checks. This includes inspecting the data for accuracy, and configurations that exceed space or performance limits.

The report is double week, so it requires 8 typed pages in which you need to clearly explain your design and calculation/analysis in detail and generated data structures. Hence, you do not need to submit a report every week but second week.

### Weeks 6 and 7

Teams will transition from dataset preparation to building and training machine learning model(s). The cleaned and structured dataset from Weeks 4 and 5 will be loaded into a preferred platform, such as Python (using machine learning frameworks like PyTorch) or MATLAB. The initial focus should be on selecting baseline machine learning algorithms/methods that can handle both regression and classification tasks, such as artificial neural networks. The machine learning model will aim to predict key outcomes, including the minimum safety factors, link dimensions, design space for the slider crank mechanism, and ideal quick return ratio for reducing the torque at the crank so that it minimizes the motor power required. Teams will split their data into training, validation, and testing subsets. Hyperparameter tuning should be performed to optimize performance, and results will be evaluated using metrics such as R², root mean square error (RMSE), loss function prediction over iteration. At the end of these weeks, each team should have at least one validated baseline machine learning model that highlights the model's predictive reliability. Teams will submit an 8-page report to explain the framework of their machine learning model, training, validation and prediction results.

### Week 8

The validated machine learning models will be applied for optimization. Teams will use their models as surrogates to explore the design space efficiently to identify slider crank configurations that balance size, strength, and manufacturability. Optimization objectives should include minimizing the size and weight of the mechanism, minimizing the motor power required while meeting design targets. The best candidate configurations identified by the machine learning model should be cross-checked with the Excel/Matlab tool to ensure compliance and accuracy. Teams should also start generating visual insights, such as sensitivity plots and parameter correlation maps, to better understand the influence of design parameters. By the end of this week, each team should have a short list of optimized designs, supported by both ML predictions and Excel tool ready for detailed CAD modeling. A four-page progress report will be submitted.

### Week 9

Teams will begin developing a detailed CAD model of the optimized configuration identified as a nominal design. Using software such as SolidWorks, each team will create precise 3D models of the slider crank mechanism components, including links, pins, bearings, housings, and covers based on the dimensions and parameters validated against the Excel tool and the machine learning outputs. Attention should be paid to maintaining accuracy in geometry, and tolerances. A four-page progress report will be submitted.

### Week 10

This week could be dedicated to the manufacturing and assembly of final CAD design for a scaled 3D-printed functional prototype. You may have the surprise that some component would not fit, there is a chance to adjust some of the components to prove feasibility of your concept. It is important that before you start printing the components to make sure that the parts are not too many or require too accurate dimensions. Please learn about the capability of the 3D printers in the lab before you start the design. Also, find and carefully read the manual of the 3D printer. Also, please make sure that your design would not require a week of printing. Please note that the software used for printing will indicate the duration of printing for a specific component. As a rule of the thumb, the small scale of parts should be built and assembled to reduce material and built time. Such aspects will be taken into consideration in the evaluation of the project. The final assembly should be carried out. One sole report of four pages or less is required on fabrication and assembly.

### Week 11

This is the last week of the project. The project write-up should be started by this week. One aspect that needs to be addressed this week is to review the lesson learned. As you know, this project is in the curriculum to prepare you for the capstone project. The capstone project follows more or less same kind of sequence, and it has very strict deadlines. This project is designed to challenge students in areas such as mechanical design, new design approaches, and user-centered design, providing them with practical skills and experience in creating a versatile and innovative design solution. A four-page progress report will be submitted.

Interesting enough is the fact that the teams as formed by the tutors in MECH 390 are usually same in the capstone project course MECH 490 where teams are formed by the initiative of the students. People who work together for some time come along and find easier to collaborate as the team members know each other and came to know their strengths and weaknesses. More or less, same things happen in the company where you will commence your professional career at the end of the program. You need to use some time to evaluate your design evolution – if any – during the course. It is extremely important to discuss the challenges that you encountered and how you addressed them if you did. Look back and evaluate if you did learn and what you did learn during the project period – classes, tutorial and individual work. You may comment what was right and what was not. This is not the class evaluation but the "learned lesson" analysis time. Try to look back and see what should make this project a better experience. Also, do not forget to fill in the statement of work – in which each team member must sign in with his agreed-by-everyone in the team his or her contribution.

You should finalize the write-up of the project report and build the required submission file. The project deliverables (a report/PowerPoint presentation and all model and calculation files including generated dataset, Excel tool and code scripts) will be submitted before the presentation which is tentatively scheduled on first or second week of April on the last day of class (9:00 am – 4:30 pm). The structure of the report is recommended to be as it follows.

Good luck with the project!

---

## Project Report Structure

The main project must include the following items in the report:

- **Cover page** with the title of the project, the team name and the members of the team with their ID numbers. You must write the following statement: *"We certify that this submission is the original work of members of the group and meets the Faculty's Expectations of Originality"*, with the signatures of all the team members and the date.
- **Abstract**
- **Table of contents**
- **List of figures and tables**
- **List of symbols and abbreviations** – if necessary
- **Introduction:** here the rationale of the work, the objectives and the targeted measurable scope of the design should be described.
- **Design Methodology:** This main section should cover:
  1. Identification of design needs and functional requirements.
  2. Design requirements and constraints, including envelope dimensions, range, speeds, and standard safety factors.
  3. Project planning and workflow.
  4. Development of the Excel-based design tool.
  5. Generation of the DOE dataset and data cleaning process.
  6. Training, validation, and testing of the machine learning model.
  7. Optimization strategies and rationale for selecting the final nominal design configuration.
  8. CAD modeling, simulation, and validation steps.
- **Results and Discussion:** Summarize analytical results, design, optimization outcomes, key performance metrics, and prototype observations. Include insights from sensitivity analyses and cross-validation.
- **Conclusions and Lessons Learned:** Discuss the main findings, how design met the objectives, and challenges faced. Each team member should reflect briefly on their individual learning experience and contributions.
- **References:** List all sources including AGMA standards, academic articles, textbooks, and software tools.
- **Appendices:** Include supporting documents such as weekly progress reports, detailed calculations, SolidWorks drawings, CAD models, code scripts, BOMs, DOE datasets, and any additional data supporting the design process.

> According to the recommendations, the main part of the report excluding Appendices should not exceed **40 pages**. The last page will include a statement of all four (or five) team members who need to declare that the work is original – not copied. Meanwhile, a distribution of the work carried out by each team member must be clearly shown. The sum of all percentile contributions will be 100%. Each team member needs to sign that he/she agrees with the statement. The statement page is extremely important in the report.

- **Prepare the presentation** – assign tasks to the team members (consider a 3 minutes talk per team member). Practice the presentation and get ready to deliver it (during the last day of class).
