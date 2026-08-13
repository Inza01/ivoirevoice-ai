"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { Brand } from "@/components/layout/brand";
import { Icon } from "@/components/ui/icon";
import { useI18n } from "@/i18n/provider";
import { getLanguageName } from "@/lib/languages/registry";

const navigation = [
  ["/", "home"],
  ["/transcribe", "transcribe"],
  ["/translate", "translate"],
  ["/learn", "learn"],
  ["/practice", "practice"],
  ["/community", "community"],
] as const;

export function Navbar() {
  const pathname = usePathname();
  const { locale, messages, setLocale } = useI18n();
  const [open, setOpen] = useState(false);
  const menuButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
        menuButtonRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open]);

  return (
    <header className="site-header">
      <div className="nav-shell">
        <Brand homeLabel={messages.navigation.home} />
        <button
          aria-controls="primary-navigation"
          aria-expanded={open}
          aria-label={open ? messages.navigation.closeMenu : messages.navigation.openMenu}
          className="menu-button"
          onClick={() => setOpen((value) => !value)}
          ref={menuButtonRef}
          type="button"
        >
          <Icon name={open ? "x" : "menu"} />
        </button>
        <nav
          aria-label={messages.navigation.primaryNavigation}
          className={open ? "primary-nav is-open" : "primary-nav"}
          id="primary-navigation"
        >
          <ul>
            {navigation.map(([href, label]) => {
              const isCurrent = href === "/" ? pathname === href : pathname.startsWith(href);
              return (
                <li key={href}>
                  <Link
                    aria-current={isCurrent ? "page" : undefined}
                    href={href}
                    onClick={() => setOpen(false)}
                  >
                    {messages.navigation[label]}
                  </Link>
                </li>
              );
            })}
          </ul>
          <div className="nav-actions">
            <label className="locale-control">
              <span className="visually-hidden">{messages.navigation.interfaceLanguage}</span>
              <Icon name="globe" />
              <select
                aria-label={messages.navigation.interfaceLanguage}
                onChange={(event) => setLocale(event.target.value === "en" ? "en" : "fr")}
                value={locale}
              >
                <option value="fr">{getLanguageName("fr", locale)}</option>
                <option value="en">{getLanguageName("en", locale)}</option>
              </select>
            </label>
            <Link className="profile-link" href="/profile" onClick={() => setOpen(false)}>
              <Icon name="profile" />
              <span>{messages.navigation.profile}</span>
            </Link>
          </div>
        </nav>
      </div>
    </header>
  );
}
