(() => {
  "use strict";

  const PRIMARY_STORAGE_KEY = "cleverly.sidebar.primary.collapsed";
  const SECONDARY_STORAGE_KEY = "cleverly.sidebar.secondary.collapsed";

  function onReady(callback) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", callback, { once: true });
      return;
    }
    callback();
  }

  function readSessionPreference(key) {
    try {
      return window.sessionStorage.getItem(key) === "true";
    } catch (_error) {
      return false;
    }
  }

  function writeSessionPreference(key, value) {
    try {
      window.sessionStorage.setItem(key, String(value));
    } catch (_error) {
      // The controls still work when session storage is unavailable.
    }
  }

  function createPanelGlyph(side) {
    const namespace = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(namespace, "svg");
    const frame = document.createElementNS(namespace, "rect");
    const panel = document.createElementNS(namespace, "rect");

    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("width", "18");
    svg.setAttribute("height", "18");
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("focusable", "false");
    frame.setAttribute("x", "3");
    frame.setAttribute("y", "4");
    frame.setAttribute("width", "18");
    frame.setAttribute("height", "16");
    frame.setAttribute("rx", "2");
    frame.setAttribute("fill", "none");
    frame.setAttribute("stroke", "currentColor");
    frame.setAttribute("stroke-width", "2");
    panel.setAttribute("x", side === "primary" ? "4" : "14");
    panel.setAttribute("y", "5");
    panel.setAttribute("width", "6");
    panel.setAttribute("height", "14");
    panel.setAttribute("fill", "currentColor");
    svg.append(frame, panel);
    return svg;
  }

  function createSidebarControls(side, sidebar, storageKey) {
    const sidebarName = side === "primary" ? "site navigation" : "On this page";
    const collapsedClass = `cleverly-${side}-sidebar-collapsed`;
    const hide = document.createElement("button");
    const show = document.createElement("button");

    hide.type = "button";
    hide.className = `cleverly-sidebar-hide cleverly-sidebar-hide--${side}`;
    hide.setAttribute("aria-label", `Hide ${sidebarName}`);
    hide.title = `Hide ${sidebarName}`;
    hide.append(createPanelGlyph(side));

    show.type = "button";
    show.className = `cleverly-sidebar-tab cleverly-sidebar-tab--${side}`;
    show.setAttribute("aria-label", `Show ${sidebarName}`);
    show.title = `Show ${sidebarName}`;
    show.append(createPanelGlyph(side));

    [hide, show].forEach((control) => {
      control.setAttribute("aria-controls", sidebar.id);
    });

    // Each state has its own control, so no element has to change its own
    // appearance. FontAwesome replaces any node carrying an fa- class, which
    // silently detaches a class name the script means to keep updating.
    const reflectState = (collapsed) => {
      document.documentElement.classList.toggle(collapsedClass, collapsed);
      hide.setAttribute("aria-expanded", String(!collapsed));
      show.setAttribute("aria-expanded", String(!collapsed));
    };

    const setCollapsed = (collapsed) => {
      reflectState(collapsed);
      writeSessionPreference(storageKey, collapsed);
      window.dispatchEvent(new Event("cleverly:layout-change"));
      window.requestAnimationFrame(() => {
        (collapsed ? show : hide).focus();
      });
    };

    reflectState(readSessionPreference(storageKey));
    hide.addEventListener("click", () => setCollapsed(true));
    show.addEventListener("click", () => setCollapsed(false));
    return { hide, show };
  }

  function mountSidebarControls(side, sidebar, storageKey, before) {
    const controls = createSidebarControls(side, sidebar, storageKey);
    const header = document.createElement("div");
    header.className = `cleverly-sidebar-panel-header cleverly-sidebar-panel-header--${side}`;
    header.append(controls.hide);
    if (before) {
      before.before(header);
    } else {
      sidebar.prepend(header);
    }
    document.body.append(controls.show);
  }

  function initializeSidebarControls() {
    const primarySidebar = document.getElementById("pst-primary-sidebar");
    const secondarySidebar = document.getElementById("pst-secondary-sidebar");

    if (primarySidebar) {
      mountSidebarControls("primary", primarySidebar, PRIMARY_STORAGE_KEY, null);
    }

    if (secondarySidebar) {
      mountSidebarControls(
        "secondary",
        secondarySidebar,
        SECONDARY_STORAGE_KEY,
        secondarySidebar.querySelector(".sidebar-secondary-items"),
      );
    }
  }

  function slugify(value) {
    return value
      .toLowerCase()
      .normalize("NFKD")
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "");
  }

  function rowSlug(row, label) {
    const studyLink = Array.from(row.querySelectorAll("a.reference.internal")).find(
      (link) => link.getAttribute("href")?.includes(".html"),
    );
    if (!studyLink) {
      return slugify(label);
    }
    const path = studyLink.getAttribute("href").split("#", 1)[0];
    return slugify(path.split("/").pop().replace(/\.html$/, ""));
  }

  function initializeValidationGrid() {
    const section = document.getElementById("implementation-validation-grid");
    const container = section?.querySelector(".pst-scrollable-table-container");
    const table = container?.querySelector("table");
    if (!section || !container || !table || !table.tBodies.length) {
      return;
    }

    section.classList.add("cleverly-validation-grid");
    const rows = Array.from(table.tBodies[0].rows);
    const headers = Array.from(table.tHead?.rows[0]?.cells || []).map((cell) =>
      cell.textContent.trim(),
    );

    const topScroller = document.createElement("div");
    const topScrollerTrack = document.createElement("div");
    topScroller.className = "cleverly-grid-top-scroll";
    topScroller.tabIndex = 0;
    topScroller.setAttribute("role", "region");
    topScroller.setAttribute("aria-label", "Horizontal scroll for validation grid");
    topScrollerTrack.className = "cleverly-grid-top-scroll__track";
    topScroller.append(topScrollerTrack);
    container.before(topScroller);

    let syncingScroll = false;
    const syncScroll = (source, target) => {
      if (syncingScroll || target.scrollLeft === source.scrollLeft) {
        return;
      }
      syncingScroll = true;
      target.scrollLeft = source.scrollLeft;
      window.requestAnimationFrame(() => {
        syncingScroll = false;
      });
    };
    topScroller.addEventListener("scroll", () => syncScroll(topScroller, container));
    container.addEventListener("scroll", () => syncScroll(container, topScroller));

    const popover = document.createElement("div");
    const popoverClose = document.createElement("button");
    const popoverContent = document.createElement("div");
    popover.id = "cleverly-grid-cell-popover";
    popover.className = "cleverly-grid-cell-popover";
    popover.hidden = true;
    popover.setAttribute("role", "dialog");
    popover.setAttribute("aria-modal", "false");
    popoverClose.type = "button";
    popoverClose.className = "cleverly-grid-cell-popover__close";
    popoverClose.setAttribute("aria-label", "Close full cell content");
    popoverClose.innerHTML = '<span class="fa-solid fa-xmark" aria-hidden="true"></span>';
    popoverContent.className = "cleverly-grid-cell-popover__content";
    popover.append(popoverClose, popoverContent);
    document.body.append(popover);

    let activeCell = null;
    let closeTimer = null;
    let suppressFocusOpen = false;

    const cancelClose = () => {
      if (closeTimer !== null) {
        window.clearTimeout(closeTimer);
        closeTimer = null;
      }
    };

    const positionPopover = () => {
      if (!activeCell || popover.hidden) {
        return;
      }
      const gap = 8;
      const cellRect = activeCell.getBoundingClientRect();
      const popoverRect = popover.getBoundingClientRect();
      const maxLeft = Math.max(gap, window.innerWidth - popoverRect.width - gap);
      const left = Math.min(Math.max(cellRect.left, gap), maxLeft);
      let top = cellRect.bottom + gap;
      if (top + popoverRect.height > window.innerHeight - gap) {
        top = cellRect.top - popoverRect.height - gap;
      }
      top = Math.min(
        Math.max(top, gap),
        Math.max(gap, window.innerHeight - popoverRect.height - gap),
      );
      popover.style.left = `${left}px`;
      popover.style.top = `${top}px`;
    };

    const closePopover = ({ restoreFocus = false } = {}) => {
      cancelClose();
      if (!activeCell) {
        return;
      }
      const previousCell = activeCell;
      previousCell.setAttribute("aria-expanded", "false");
      activeCell = null;
      popover.hidden = true;
      popoverContent.replaceChildren();
      if (restoreFocus) {
        suppressFocusOpen = true;
        previousCell.focus({ preventScroll: true });
        window.requestAnimationFrame(() => {
          suppressFocusOpen = false;
        });
      }
    };

    const scheduleClose = () => {
      cancelClose();
      closeTimer = window.setTimeout(() => {
        if (
          activeCell &&
          !activeCell.matches(":hover") &&
          !activeCell.contains(document.activeElement) &&
          !popover.matches(":hover") &&
          !popover.contains(document.activeElement)
        ) {
          closePopover();
        }
      }, 120);
    };

    const openPopover = (cell, { moveFocus = false } = {}) => {
      cancelClose();
      if (!cell.classList.contains("cleverly-grid-cell--truncated")) {
        return;
      }
      if (activeCell && activeCell !== cell) {
        activeCell.setAttribute("aria-expanded", "false");
      }
      activeCell = cell;
      const content = cell.querySelector(".cleverly-grid-cell-content");
      popoverContent.replaceChildren(...Array.from(content.childNodes).map((node) => node.cloneNode(true)));
      const rowLabel = cell.parentElement.cells[0].textContent.trim();
      const columnLabel = headers[cell.cellIndex] || `Column ${cell.cellIndex + 1}`;
      popover.setAttribute("aria-label", `${rowLabel}: ${columnLabel}`);
      cell.setAttribute("aria-expanded", "true");
      popover.hidden = false;
      positionPopover();
      if (moveFocus) {
        const focusTarget = popoverContent.querySelector("a, button") || popoverClose;
        focusTarget.focus();
      }
    };

    popoverClose.addEventListener("click", () => closePopover({ restoreFocus: true }));
    popover.addEventListener("mouseenter", cancelClose);
    popover.addEventListener("mouseleave", scheduleClose);
    popover.addEventListener("focusin", cancelClose);
    popover.addEventListener("focusout", scheduleClose);

    const gridCells = Array.from(table.tBodies[0].querySelectorAll("td"));
    gridCells.forEach((cell) => {
      const content = document.createElement("div");
      content.className = "cleverly-grid-cell-content";
      content.append(...Array.from(cell.childNodes));
      cell.append(content);
      cell.setAttribute("aria-controls", popover.id);
      cell.setAttribute("aria-expanded", "false");
      cell.addEventListener("mouseenter", () => openPopover(cell));
      cell.addEventListener("mouseleave", scheduleClose);
      cell.addEventListener("focusin", () => {
        if (!suppressFocusOpen) {
          openPopover(cell);
        }
      });
      cell.addEventListener("focusout", scheduleClose);
      cell.addEventListener("click", (event) => {
        if (!event.target.closest("a")) {
          openPopover(cell);
        }
      });
      cell.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          openPopover(cell, { moveFocus: true });
        } else if (event.key === "Escape" && activeCell === cell) {
          event.preventDefault();
          closePopover({ restoreFocus: true });
        }
      });
    });

    const refreshGridLayout = () => {
      const scrollbarGutter = topScroller.clientWidth - container.clientWidth;
      topScrollerTrack.style.width = `${container.scrollWidth + scrollbarGutter}px`;
      gridCells.forEach((cell) => {
        const content = cell.querySelector(".cleverly-grid-cell-content");
        const truncated = content.scrollHeight > content.clientHeight + 1;
        cell.classList.toggle("cleverly-grid-cell--truncated", truncated);
        cell.tabIndex = truncated ? 0 : -1;
        if (!truncated && activeCell === cell) {
          closePopover();
        }
      });
      positionPopover();
    };

    let refreshFrame = null;
    const scheduleRefresh = () => {
      if (refreshFrame !== null) {
        window.cancelAnimationFrame(refreshFrame);
      }
      refreshFrame = window.requestAnimationFrame(() => {
        refreshFrame = null;
        refreshGridLayout();
      });
    };

    const rowLinks = new Map();
    const usedIds = new Set();
    rows.forEach((row, index) => {
      const label = row.cells[0].textContent.trim();
      const baseId = `study-${rowSlug(row, label) || index + 1}`;
      let id = baseId;
      let suffix = 2;
      while (usedIds.has(id)) {
        id = `${baseId}-${suffix}`;
        suffix += 1;
      }
      usedIds.add(id);
      row.id = id;
      row.tabIndex = -1;
      row.dataset.studyLabel = label;
    });

    const secondaryItems = document.querySelector(
      "#pst-secondary-sidebar .sidebar-secondary-items",
    );
    let activeRow = null;
    const setActiveRow = (row) => {
      if (activeRow === row) {
        return;
      }
      activeRow?.classList.remove("cleverly-grid-active-row");
      const previousLink = activeRow ? rowLinks.get(activeRow) : null;
      previousLink?.removeAttribute("aria-current");
      activeRow = row;
      activeRow?.classList.add("cleverly-grid-active-row");
      const activeLink = activeRow ? rowLinks.get(activeRow) : null;
      activeLink?.setAttribute("aria-current", "location");
      activeLink?.scrollIntoView({ block: "nearest" });
    };

    const scrollToRow = (row, { focus = false, smooth = false } = {}) => {
      const headerHeight = table.tHead?.getBoundingClientRect().height || 0;
      container.scrollTo({
        top: Math.max(0, row.offsetTop - headerHeight),
        behavior:
          smooth && !window.matchMedia("(prefers-reduced-motion: reduce)").matches
            ? "smooth"
            : "auto",
      });
      setActiveRow(row);
      if (focus) {
        row.focus({ preventScroll: true });
      }
    };

    if (secondaryItems) {
      const sidebarItem = document.createElement("div");
      const nav = document.createElement("nav");
      const heading = document.createElement("h3");
      const list = document.createElement("ul");
      sidebarItem.className =
        "sidebar-secondary-item cleverly-grid-study-navigation";
      nav.setAttribute("aria-label", "Validation studies");
      heading.textContent = "Studies";
      list.className = "cleverly-grid-study-navigation__list";
      rows.forEach((row) => {
        const item = document.createElement("li");
        const link = document.createElement("a");
        link.href = `#${row.id}`;
        link.textContent = row.dataset.studyLabel;
        link.addEventListener("click", (event) => {
          event.preventDefault();
          window.history.pushState(null, "", link.href);
          scrollToRow(row, { focus: true, smooth: true });
        });
        item.append(link);
        list.append(item);
        rowLinks.set(row, link);
      });
      nav.append(heading, list);
      sidebarItem.append(nav);
      secondaryItems.prepend(sidebarItem);
    }

    let activeFrame = null;
    const updateActiveRow = () => {
      if (activeFrame !== null) {
        return;
      }
      activeFrame = window.requestAnimationFrame(() => {
        activeFrame = null;
        const headerHeight = table.tHead?.getBoundingClientRect().height || 0;
        const position = container.scrollTop + headerHeight + 2;
        let current;
        if (
          container.scrollTop + container.clientHeight >=
          container.scrollHeight - 2
        ) {
          current = rows[rows.length - 1];
        } else {
          current = rows[0];
          for (const row of rows) {
            if (row.offsetTop > position) {
              break;
            }
            current = row;
          }
        }
        setActiveRow(current);
        positionPopover();
      });
    };
    container.addEventListener("scroll", updateActiveRow, { passive: true });

    const handleFragment = () => {
      const id = decodeURIComponent(window.location.hash.slice(1));
      const row = rows.find((candidate) => candidate.id === id);
      if (row) {
        scrollToRow(row);
      }
    };
    window.addEventListener("hashchange", handleFragment);
    window.addEventListener("popstate", handleFragment);
    window.addEventListener("resize", scheduleRefresh, { passive: true });
    window.addEventListener("cleverly:layout-change", scheduleRefresh);
    window.addEventListener("scroll", positionPopover, { capture: true, passive: true });
    document.addEventListener("pointerdown", (event) => {
      if (
        activeCell &&
        !activeCell.contains(event.target) &&
        !popover.contains(event.target)
      ) {
        closePopover();
      }
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && activeCell) {
        event.preventDefault();
        closePopover({ restoreFocus: true });
      }
    });

    if ("ResizeObserver" in window) {
      const observer = new ResizeObserver(scheduleRefresh);
      observer.observe(container);
      observer.observe(table);
    }

    scheduleRefresh();
    setActiveRow(rows[0]);
    window.requestAnimationFrame(handleFragment);
  }

  onReady(() => {
    initializeSidebarControls();
    initializeValidationGrid();
  });
})();
