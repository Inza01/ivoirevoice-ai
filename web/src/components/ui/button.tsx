import Link from "next/link";
import type { ButtonHTMLAttributes, ReactNode } from "react";

type SharedProps = {
  children: ReactNode;
  className?: string;
  href?: string;
};

type ButtonProps = SharedProps &
  Pick<ButtonHTMLAttributes<HTMLButtonElement>, "aria-label" | "disabled" | "onClick" | "type">;

function action(
  variant: "button-primary" | "button-secondary",
  { children, className = "", href, ...props }: ButtonProps,
) {
  const classes = `${variant} ${className}`.trim();

  if (href) {
    return (
      <Link className={classes} href={href}>
        {children}
      </Link>
    );
  }

  return (
    <button className={classes} type={props.type ?? "button"} {...props}>
      {children}
    </button>
  );
}

export function PrimaryButton(props: ButtonProps) {
  return action("button-primary", props);
}

export function SecondaryButton(props: ButtonProps) {
  return action("button-secondary", props);
}
