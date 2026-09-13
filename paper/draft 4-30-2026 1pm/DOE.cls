% Example calls:
% 
%	\documentclass{NSF}
%   \documentclass[draft,pgs,endfloat,double,edit]{ARO}
%
% Options:
%
%	draft	   insert placeholders for figures; enables a quick compile
% pgs      add a header with the page number; helpful for counting page length
% endfloat push figures to end, one per page, top of page; helpful for editing
% double   double line spacing, helpful for by-hand editing
% edit     narrow margins, helpful for by-hand editing
%
% 2020/12/05.  Add double option, edit option. 
 
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{DOE}[2026/04/29 Class for preparing DOE grants]
\LoadClass[11pt]{article} 

% FONTS
% =====
%
%% For LaTeX available fonts, see https://en.wikibooks.org/wiki/LaTeX/Fonts

% jam99 04/29/26
% Times New Roman in OVERLEAF
% https://www.overleaf.com/learn/latex/XeLaTeX#The_basics:_Times_New_Roman
%
%
% 
\RequirePackage{anyfontsize}  % <== jam99 04/29/26
\RequirePackage{fontspec}     % <== jam99 04/29/26
\setmainfont{Times New Roman} % <== jam99 04/29/26
\AtBeginDocument{\fontsize{11bp}{13.2bp}\selectfont}  % <== jam99 04/29/26: now 11pt in Adobe

% FIGURES
% =======
%
% Use the standard graphicx package for figure inclusion.  Use the caption package
% to get small text for figure captions.  Add a line below the figure caption.  Finally, 
% allow full-page figures.  See the discussion at 
% http://dcwww.camd.dtu.dk/~schiotz/comp/LatexTips/LatexTips.html#figplacement
%
% It is important to import the rotating package before the caption package

\RequirePackage{graphicx}
\RequirePackage[footnotesize, %small, % normalsize
	labelfont=sc,
	% labelsep=endash,
	singlelinecheck=false,
	belowskip=-1.0ex]{caption}

% \DeclareCaptionFormat{myformat}{#1#2#3\vspace{-1ex}\hrulefill}
\DeclareCaptionFormat{myformat}{#1#2#3\vspace{0ex}}
\captionsetup[figure]{format=myformat}


% FIGURE PLACEMENT
% ================
%
% Unless these settings are set JUST SO, using the super option
% with the natbib package puts all the figures at the end of 
% the document -- extremely perplexing!

\renewcommand{\topfraction}{.90}
\renewcommand{\bottomfraction}{.90}
\renewcommand{\textfraction}{.12}
\renewcommand{\floatpagefraction}{.90}
\renewcommand{\dbltopfraction}{.90}
\renewcommand{\dblfloatpagefraction}{.90}
\setcounter{topnumber}{9}
\setcounter{bottomnumber}{9}
\setcounter{totalnumber}{20}
\setcounter{dbltopnumber}{9}

% MATH and UNITS
% ==============

\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{bm}
% see https://tex.stackexchange.com/questions/403693/use-text-font-in-siunitx-instead-of-math-font
\RequirePackage[
  mode=match,
  propagate-math-font=true,
  reset-math-version=false,
  reset-text-family=false,
  reset-text-series=false,
  text-family-to-math=true,
  text-series-to-math=true
]{siunitx}
%\RequirePackage{siunitx}
%\sisetup{ 
%    detect-all = true,
%    detect-display-math = true,
%    separate-uncertainty=true
%}
\DeclareSIUnit\photon{photon}
\RequirePackage{cases} 
\RequirePackage{mathtools}

% TABLES
% ======

\RequirePackage{dcolumn}
\RequirePackage{booktabs}
\RequirePackage{tabularx} 

% MARGINS and HEADERS
% ===================
%
% For an NIH R01 grant, the margins can be as small as 0.50 inches.
% The header and footer should be totally empty -- not even page
% numbers --- since the header and footer will be added when you import
% your pdf into the grants.gov pdf template.
%
% For an ARO grant, the margins need to be 1.00 inch.  Make the default
% top and bottom blank.  The "pgs" option below will add a line at 
% top and bottom and a page number.  With a one inch margin, I find
% I need to tweak footskip from 2ex to 4ex.

\RequirePackage[letterpaper,  
	top=0.75in,
	bottom=0.75in,
	left=1.00in,
	right=1.00in,
	includehead,
	includefoot,
	headsep=2ex,
	footskip=4ex]{geometry}
 
\RequirePackage{fancyhdr}
\renewcommand{\headrulewidth}{0.0pt} 
\renewcommand{\footrulewidth}{0.0pt}
\pagestyle{fancy}

	\lhead{}
	\chead{}
	\rhead{}
	\lfoot{}
	\cfoot{}  
	\cfoot{}
	\rfoot{}

% OTHER PACKAGES
% ==============

\RequirePackage{overpic}
\RequirePackage{wrapfig}
\RequirePackage[utf8]{inputenc}
\RequirePackage{etoolbox}

% -----------
% Option: pgs
% -----------
%
% Add a line across the top and bottom of the page and
% add a page number at the bottom of the page.


\DeclareOption{pgs}{
	\renewcommand{\headrulewidth}{0.4pt} 
	\renewcommand{\footrulewidth}{0.4pt}
	\cfoot{\thepage}
}

% ----------------
% Option: endfloat
% ----------------

\newif\if@loadendfloat@  
\DeclareOption{endfloat}{\@loadendfloat@true}

% --------------
% Option: double
% --------------

\newif\if@loaddouble@  
\DeclareOption{double}{\@loaddouble@true}

% --------------
% Option: edit
% --------------

\newif\if@loadedit@  
\DeclareOption{edit}{\@loadedit@true}

% -------------
% Option: draft
% -------------
%
% Pass this option to the article class AND
% to the graphicx package.  The \setkeys{}
% is a kludge.  See
%
% http://tex.stackexchange.com/questions/9751/draft-mode-after-include-statements

\DeclareOption{draft}{
  \PassOptionsToClass{\CurrentOption}{article}
  \setkeys{Gin}{draft}
}

% ---------------
% Process options
% ---------------

\ProcessOptions\relax

% EDIT
% ====
%
% Set narrow margins to leave room for by-hand edits

\if@loadedit@	
	\geometry{top=0.25in, bottom=0.25in, left=0.25in, right=3.75in}
	\setlength\headwidth{8in}
	\RequirePackage[en-GB, showdow=true, showseconds=false]{datetime2}
	\chead{\DTMtoday{} at \DTMcurrenttime{}}
\fi

% ENDFLOAT
% ========
% 
% If the endfloat option is invoked, then place figures and 
% tables at the end.  We do not want a list of figures.  We
% want one figure per page, and we want each figure to appear
% at the top of the page.  
%
% This is achieved by modifying \efloatseparator.  Endfloat 
% places \efloatseparator after each float. By default it is
% defined to be \clearpage.  Redefine \efloatseparator so that
% the page is filled to the bottom before the \clearpage.  
% Now each figure appears at the top of its page instead of in
% the center of its page. 

\if@loadendfloat@
  \RequirePackage[nolists,nomarkers]{endfloat}
  \renewcommand{\efloatseparator}{\mbox{} \vfill \clearpage}
\fi

% HYPERLINKS
% ==========
%
% Change the color of hyperlinks to black to conform with
% NIH rules -- no COLOR text.

\RequirePackage[colorlinks=true,
	citecolor=black,
	linkcolor=black,
	urlcolor=black,
	pagebackref=false]{hyperref}

% SPACINGS
% ========
%
% Invoking \sloppy helps stop text from spilling into the margins.  The default
% LaTeX hyphenation gives too many word breaks; tweak the hyphenation.  Finally,
% set the paragraph indent and spacing.  Use "ex" for a vertical space in units of
% the font size; use "em" for a horizontal space in units of font size.

\sloppy
\lefthyphenmin=3
\righthyphenmin=4
\setlength\parindent{2em} 
\setlength\parskip{0.5ex} 

% DOUBLE
% ======

\if@loaddouble@
	\RequirePackage[doublespacing]{setspace}
\fi

% SECTIONING
% ==========
% 
% Redefine the commands for
%
% 	\section{}       Large bold sans-serif font, no label, underline, CR; 
%                     e.g., Research Strategy
%
%   \subsection{}    large bold sans-serif font, label with letter, CR; 
%                     e.g., A. Significance, B. Innovation
%
%   \subsubsection{} normal bold sans-serif font, no label, CR; 
%                     e.g., Overview of the approach, Aim 1
%
%   \paragraph{}     normal bold sans-serif font, long dash, no CR; 
%                     e.g., a Sub-aim
%
% where CR is shorthand for carriage return.  In each case I have trimmed the vertical spacing
% before and after the sectioning commands.  For an ARO proposal, do not reset the page 
% number at the start of each section.

\RequirePackage{titlesec} 

% \titleformat
% {} the sectioning command to be redefined
% {} the format to be applied to the whole title label and text
% {} the label; leave it empty if there is no section label
% {} horiz separation between label and title body; must be a length, i.e. 0em
% {} code preceding the title body; could be \setcounter{page}{1}
% [\titlerule] % inserts a horizontal line after the heading

% \titlespacing*
% {\section} % the sectioning command whose spacing is to be adjusting
% {0em} increases the left margin
% {0ex} the vertical space before the title
% {0.5ex} the separation between title and tex

% jam99 04/29/26: Rewrite the sectioning commands
% 
% \renewcommand\thesection{\Roman{section}}
% \titlespacing*{\section}{0em}{0ex}{0.5ex} 
% \titleformat{\section}
% 	{\Large\bfseries\sffamily\raggedright}{\Roman{section}.}
% 	{0.5em}{}[\titlerule]

\renewcommand\thesection{\Roman{section}} % Set numbering to Roman

\titleformat{\section}[runin]                     % Use 'runin' for inline text
	{\normalsize\bfseries\sffamily\raggedright}   % Format of the label and title
	{\thesection.}                                % Label format (Roman numeral + dot)
	{0.5em}                                       % Space between label and title
	{}                                            % Code preceding title body
	[{.}]                                         % Punctuation after title + space

\titlespacing*{\section} % Define the section spacing
	{0em}                % Left margin
	{0ex}                % Space before
	{0.5em}              % Space after (between title dot and body text)

\renewcommand\thesubsection{\Roman{section}.\Alph{subsection}} 
\titlespacing*{\subsection}{0em}{0ex}{0ex}
\titleformat{\subsection}[runin]
{\normalsize\bfseries}{\Alph{subsection}.}
{0.5em}{}[\mdseries\rmfamily \hspace{0.5em}] % jam99 remove period
        % [\mdseries\rmfamily. \hspace{0.5em}]
        % [\normalsize\bfseries \hspace{0.5em}]
	% [\hspace{0em} \mdseries\rmfamily --- \hspace{0em}]

\renewcommand\thesubsubsection{\Roman{section}.\Alph{subsection}.\arabic{subsubsection}} 
\titlespacing*{\subsubsection}{0em}{0ex}{0ex}
\titleformat{\subsubsection}[runin]
	{\normalsize\bfseries}{\Alph{subsection}.\arabic{subsubsection}.}
	{0.5em}{}  [\mdseries\rmfamily \hspace{0.5em}] % jam99 remove period
	% [\mdseries\rmfamily. \hspace{0.5em}]
	% [\hspace{0em} \mdseries\rmfamily \hspace{0em}] % <== jam99, 2024-02-25, for DOE.cls.  Remove period, let user do punctuation.
	% [\mdseries\rmfamily. \hspace{0.5em}]
	% [\hspace{0em} \mdseries\rmfamily --- \hspace{0em}]

%\titlespacing*{\subsubsection}{0em}{1ex}{-0.25ex}
%\titleformat{\subsubsection}{\normalsize\bfseries\sffamily}{}{0em}{}
%\newcommand{\subaim}[3]{\paragraph{#1} [Year #2 to #3] ---}

\titlespacing*{\paragraph}{0em}{0ex}{0ex}
\titleformat{\paragraph}[runin]{\normalsize\bfseries}{}{0.5em}{}[\normalsize\bfseries. \hspace{0.5em}]
\newcommand{\subpoint}[1]{\textsection \hspace{0.25em} {\bf #1} ---}
 	
% CHEMISTRY
% =========

\RequirePackage[version=4]{mhchem}

% LISTS
% =====

\RequirePackage[inline]{enumitem}
\setlist[1]{nolistsep,leftmargin=2em}
\setlist[2]{nolistsep,leftmargin=2em}

% BIBLIOGRAPHY
% ============
%
% Need the soul package for underlining, \ul{}.  I am setting 
% the citations to use superscript numbers and not bracketed 
% numbers because this cuts 6 lines from a 12 page NIH proposal
% -- significant.  If you want Physical Review Letters style
% bracketed citations, then use the following import statement
% instead:
% 
% 	\RequirePackage[sort&compress,square,numbers]{natbib}
%

\RequirePackage{soul}
\RequirePackage[super,sort&compress,numbers]{natbib}
\RequirePackage{natmove}

\def\bibsection{
	\clearpage
	\section{Cited References}
	\vspace{2ex}
	}
\setlength{\bibsep}{2ex}
% \renewcommand*{\bibfont}{\raggedright}


% NEW COMMANDS
% ============
%
% The command \extendstrategy will create new pages until the 
% page count is fifteen.  This command is handy for filling out
% the strategy section while you are writing to see how much 
% space you have to work with. 

\RequirePackage{forloop}
\newcommand{\extendstrategy}[1]{

	\newcounter{pgs}
	\newcounter{init}
	\setcounter{init}{\value{page}}
	\forloop{pgs}{\value{init}}{\value{pgs} < 16}{
		\null\newpage
	}
}

% The command \addhead will create a nice header saying, for example,
%
% 	left top: Army Research Office Proposal
% 	right top: Principal Investigator: Marohn, John A.
%
% in a distinguishing sans-serif font.  To get this header, you 
% would call
%
%	\addhead{Army Research Office Proposal}{Marohn, John A.}

\newcommand{\addhead}[2]{
	\lhead{\textsf{#1}}
	\rhead{\textsf{PI: #2}}
}

% Highlight the PI's name in bold, helpful in a collaborative
% proposal to quickly ascertain who is doing what.
% Usage: \PI{Jones} 

\newcommand{\PI}[1]{{\bf #1}}

% Two commands to help reset the page numbers, one to set the page
% numbers to be lower-case roman numerals and a second one to set
% the page number to be arabic numbers.

\newcommand{\pageresetpreamble}[0]{
	\pagenumbering{roman}
	\setcounter{page}{1}
}

\newcommand{\pageresetmain}[0]{
	\pagenumbering{arabic}
	\setcounter{page}{1}
}

% Some handy math shortcuts

\RequirePackage{mathtools}
\DeclarePairedDelimiter\abs{\lvert}{\rvert}
\renewcommand{\Re}{\operatorname{Re}}
\renewcommand{\Im}{\operatorname{Im}}
\renewcommand{\st}[1]{_{\mathrm{#1}}} 

% SUBFILES
% ========
%
% Alternatives to \include{} and input{}, as discussed here:
%
% 	https://en.wikibooks.org/wiki/LaTeX/Modular_Documents
%	http://ftp.math.purdue.edu/mirrors/ctan.org/macros/latex/contrib/subfiles/subfiles.pdf
%	http://mirror.hmc.edu/ctan/macros/latex/contrib/standalone/standalone.pdf

\RequirePackage{subfiles}
\RequirePackage{pdfpages}

% BIBENTRY
% ========

\RequirePackage{bibentry}

% Workaround for getting bibentry to play nice with 
% hyperref.  Google "bibentry Lonely \item" and find
%
% 	http://stefaanlippens.net/bibentry?page=1
%
% Beyond this, I find that bibentry and hyperref still
% interact in strange ways.  For example, you cannot
% use the  pagebackref=true option in hyperref, even
% with this fix.

\makeatletter % bibtentry & hyperref
\let\saved@bibitem\@bibitem % bibtentry & hyperref
\makeatother % bibtentry & hyperref

% FOR TESTING
% ===========

\RequirePackage{lipsum}

\endinput

	

